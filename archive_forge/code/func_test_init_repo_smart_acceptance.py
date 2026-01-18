from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_init_repo_smart_acceptance(self):
    self.setup_smart_server_with_call_log()
    self.run_bzr(['init-shared-repo', self.get_url('repo')])
    self.assertLength(11, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)