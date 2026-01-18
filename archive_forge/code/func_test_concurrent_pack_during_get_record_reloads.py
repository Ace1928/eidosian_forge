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
def test_concurrent_pack_during_get_record_reloads(self):
    tree = self.make_branch_and_tree('tree')
    with tree.lock_write():
        rev1 = tree.commit('one')
        rev2 = tree.commit('two')
        keys = [(rev1,), (rev2,)]
        r2 = repository.Repository.open('tree')
        with r2.lock_read():
            packed = False
            result = {}
            record_stream = r2.revisions.get_record_stream(keys, 'unordered', False)
            for record in record_stream:
                result[record.key] = record
                if not packed:
                    tree.branch.repository.pack()
                    packed = True
            self.assertEqual(sorted(keys), sorted(result.keys()))