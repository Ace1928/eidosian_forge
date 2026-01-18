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
def test_chk_bytes_uses_custom_btree_parser(self):
    mt = self.make_branch_and_memory_tree('test', format='2a')
    mt.lock_write()
    self.addCleanup(mt.unlock)
    mt.add([''], [b'root-id'])
    mt.commit('first')
    index = mt.branch.repository.chk_bytes._index._graph_index._indices[0]
    self.assertEqual(btree_index._gcchk_factory, index._leaf_factory)
    repo = mt.branch.repository.controldir.open_repository()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    index = repo.chk_bytes._index._graph_index._indices[0]
    self.assertEqual(btree_index._gcchk_factory, index._leaf_factory)