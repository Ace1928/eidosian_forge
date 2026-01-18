from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_from_trivial_branch_to_same_server_branch_acceptance(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    for count in range(9):
        t.commit(message='commit %d' % count)
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', self.get_url('from'), self.get_url('target')])
    self.assertLength(2, self.hpss_connections)
    self.assertLength(34, self.hpss_calls)
    self.expectFailure('branching to the same branch requires VFS access', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)