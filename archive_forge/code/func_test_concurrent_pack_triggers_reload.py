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
def test_concurrent_pack_triggers_reload(self):
    tree = self.make_branch_and_tree('tree')
    with tree.lock_write():
        rev1 = tree.commit('one')
        rev2 = tree.commit('two')
        r2 = repository.Repository.open('tree')
        with r2.lock_read():
            tree.branch.repository.pack()
            self.assertEqual({rev2: (rev1,)}, r2.get_parent_map([rev2]))