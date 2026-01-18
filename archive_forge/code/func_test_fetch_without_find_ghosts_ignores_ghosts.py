from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_fetch_without_find_ghosts_ignores_ghosts(self):
    has_ghost = self.make_repository('has_ghost', format=self.get_format())
    missing_ghost = self.make_repository('missing_ghost', format=self.get_format())

    def add_commit(repo, revision_id, parent_ids):
        repo.lock_write()
        repo.start_write_group()
        inv = inventory.Inventory(revision_id=revision_id)
        inv.root.revision = revision_id
        root_id = inv.root.file_id
        sha1 = repo.add_inventory(revision_id, inv, [])
        repo.texts.add_lines((root_id, revision_id), [], [])
        rev = _mod_revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
        rev.parent_ids = parent_ids
        repo.add_revision(revision_id, rev)
        repo.commit_write_group()
        repo.unlock()
    add_commit(has_ghost, b'ghost', [])
    add_commit(has_ghost, b'references', [b'ghost'])
    add_commit(missing_ghost, b'references', [b'ghost'])
    add_commit(has_ghost, b'tip', [b'references'])
    missing_ghost.fetch(has_ghost, b'tip')
    rev = missing_ghost.get_revision(b'tip')
    inv = missing_ghost.get_inventory(b'tip')
    self.assertRaises(errors.NoSuchRevision, missing_ghost.get_revision, b'ghost')
    self.assertRaises(errors.NoSuchRevision, missing_ghost.get_inventory, b'ghost')