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
def test_abort_write_group_does_not_raise_when_suppressed(self):
    """Similar to per_repository.test_write_group's test of the same name.

        Also requires that the exception is logged.
        """
    self.vfs_transport_factory = memory.MemoryServer
    repo = self.make_repository('repo', format=self.get_format())
    token = self._lock_write(repo).repository_token
    repo.start_write_group()
    self.get_transport('').rename('repo', 'foo')
    self.assertEqual(None, repo.abort_write_group(suppress_errors=True))
    log = self.get_log()
    self.assertContainsRe(log, 'abort_write_group failed')
    self.assertContainsRe(log, 'INFO  brz: ERROR \\(ignored\\):')
    if token is not None:
        repo.leave_lock_in_place()