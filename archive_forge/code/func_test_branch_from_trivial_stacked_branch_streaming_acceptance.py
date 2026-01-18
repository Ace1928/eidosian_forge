from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_from_trivial_stacked_branch_streaming_acceptance(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('trunk')
    for count in range(8):
        t.commit(message='commit %d' % count)
    tree2 = t.branch.controldir.sprout('feature', stacked=True).open_workingtree()
    local_tree = t.branch.controldir.sprout('local-working').open_workingtree()
    local_tree.commit('feature change')
    local_tree.branch.push(tree2.branch)
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', self.get_url('feature'), 'local-target'])
    self.assertLength(16, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)