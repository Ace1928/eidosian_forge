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
def test_inconsistency_fatal(self):
    repo = self.make_repository('repo', format='2a')
    self.assertTrue(repo.revisions._index._inconsistency_fatal)
    self.assertFalse(repo.texts._index._inconsistency_fatal)
    self.assertFalse(repo.inventories._index._inconsistency_fatal)
    self.assertFalse(repo.signatures._index._inconsistency_fatal)
    self.assertFalse(repo.chk_bytes._index._inconsistency_fatal)