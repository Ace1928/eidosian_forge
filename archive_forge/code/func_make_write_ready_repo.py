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
def make_write_ready_repo(self):
    format = self.get_format()
    if isinstance(format.repository_format, RepositoryFormat2a):
        raise TestNotApplicable('No missing compression parents')
    repo = self.make_repository('.', format=format)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    self.addCleanup(repo.abort_write_group)
    return repo