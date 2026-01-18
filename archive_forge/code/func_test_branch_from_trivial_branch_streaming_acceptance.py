from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_from_trivial_branch_streaming_acceptance(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    for count in range(9):
        t.commit(message='commit %d' % count)
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', self.get_url('from'), 'local-target'])
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
    self.assertLength(11, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)