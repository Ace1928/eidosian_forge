from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def make_broken_repository(self):
    repo = self.make_repository('broken-repo')
    cleanups = []
    try:
        repo.lock_write()
        cleanups.append(repo.unlock)
        repo.start_write_group()
        cleanups.append(repo.commit_write_group)
        inv = inventory.Inventory(revision_id=b'rev1a')
        inv.root.revision = b'rev1a'
        self.add_file(repo, inv, 'file1', b'rev1a', [])
        repo.texts.add_lines((inv.root.file_id, b'rev1a'), [], [])
        repo.add_inventory(b'rev1a', inv, [])
        revision = _mod_revision.Revision(b'rev1a', committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=[])
        repo.add_revision(b'rev1a', revision, inv)
        inv = inventory.Inventory(revision_id=b'rev1b')
        inv.root.revision = b'rev1b'
        self.add_file(repo, inv, 'file1', b'rev1b', [])
        repo.add_inventory(b'rev1b', inv, [])
        inv = inventory.Inventory()
        self.add_file(repo, inv, 'file1', b'rev2', [b'rev1a', b'rev1b'])
        self.add_file(repo, inv, 'file2', b'rev2', [])
        self.add_revision(repo, b'rev2', inv, [b'rev1a'])
        inv = inventory.Inventory()
        self.add_file(repo, inv, 'file2', b'rev1c', [])
        inv = inventory.Inventory()
        self.add_file(repo, inv, 'file2', b'rev3', [b'rev1c'])
        self.add_revision(repo, b'rev3', inv, [b'rev1c'])
        return repo
    finally:
        for cleanup in reversed(cleanups):
            cleanup()