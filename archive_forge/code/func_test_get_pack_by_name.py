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
def test_get_pack_by_name(self):
    format = self.get_format()
    tree = self.make_branch_and_tree('.', format=format)
    tree.commit('start')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    packs = tree.branch.repository._pack_collection
    packs.reset()
    packs.ensure_loaded()
    name = packs.names()[0]
    pack_1 = packs.get_pack_by_name(name)
    sizes = packs._names[name]
    rev_index = GraphIndex(packs._index_transport, name + '.rix', sizes[0])
    inv_index = GraphIndex(packs._index_transport, name + '.iix', sizes[1])
    txt_index = GraphIndex(packs._index_transport, name + '.tix', sizes[2])
    sig_index = GraphIndex(packs._index_transport, name + '.six', sizes[3])
    self.assertEqual(pack_repo.ExistingPack(packs._pack_transport, name, rev_index, inv_index, txt_index, sig_index), pack_1)
    self.assertTrue(pack_1 is packs.get_pack_by_name(name))