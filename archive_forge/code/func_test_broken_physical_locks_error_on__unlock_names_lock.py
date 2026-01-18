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
def test_broken_physical_locks_error_on__unlock_names_lock(self):
    repo = self.make_repository('.', format=self.get_format())
    repo._pack_collection.lock_names()
    self.assertTrue(repo.get_physical_lock_status())
    repo2 = repository.Repository.open('.')
    self.prepare_for_break_lock()
    repo2.break_lock()
    self.assertRaises(errors.LockBroken, repo._pack_collection._unlock_names)