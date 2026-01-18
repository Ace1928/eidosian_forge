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
def test_adding_revision_creates_pack_indices(self):
    format = self.get_format()
    tree = self.make_branch_and_tree('.', format=format)
    trans = tree.branch.repository.controldir.get_repository_transport(None)
    self.assertEqual([], list(self.index_class(trans, 'pack-names', None).iter_all_entries()))
    tree.commit('foobarbaz')
    index = self.index_class(trans, 'pack-names', None)
    index_nodes = list(index.iter_all_entries())
    self.assertEqual(1, len(index_nodes))
    node = index_nodes[0]
    name = node[1][0]
    pack_value = node[2]
    sizes = [int(digits) for digits in pack_value.split(b' ')]
    for size, suffix in zip(sizes, ['.rix', '.iix', '.tix', '.six']):
        stat = trans.stat('indices/{}{}'.format(name.decode('ascii'), suffix))
        self.assertEqual(size, stat.st_size)