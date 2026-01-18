from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_to_stacked_from_trivial_branch_streaming_acceptance(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    for count in range(9):
        t.commit(message='commit %d' % count)
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', '--stacked', self.get_url('from'), 'local-target'])
    readvs_of_rix_files = [c for c in self.hpss_calls if c.call.method == 'readv' and c.call.args[-1].endswith('.rix')]
    self.assertLength(1, self.hpss_connections)
    self.assertLength(0, readvs_of_rix_files)
    self.expectFailure('branching to stacked requires VFS access', self.assertThat, self.hpss_calls, ContainsNoVfsCalls)