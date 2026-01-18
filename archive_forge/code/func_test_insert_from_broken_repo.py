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
def test_insert_from_broken_repo(self):
    """Inserting a data stream from a broken repository won't silently
        corrupt the target repository.
        """
    broken_repo = self.make_broken_repository()
    empty_repo = self.make_repository('empty-repo')
    try:
        empty_repo.fetch(broken_repo)
    except (errors.RevisionNotPresent, errors.BzrCheckError):
        return
    empty_repo.lock_read()
    self.addCleanup(empty_repo.unlock)
    text = next(empty_repo.texts.get_record_stream([(b'file2-id', b'rev3')], 'topological', True))
    self.assertEqual(b'line\n', text.get_bytes_as('fulltext'))