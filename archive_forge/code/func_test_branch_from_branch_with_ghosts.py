from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def test_branch_from_branch_with_ghosts(self):
    self.setup_smart_server_with_call_log()
    t = self.make_branch_and_tree('from')
    for count in range(9):
        t.commit(message='commit %d' % count)
    t.set_parent_ids([t.last_revision(), b'ghost'])
    t.commit(message='add commit with parent')
    self.reset_smart_call_log()
    out, err = self.run_bzr(['branch', self.get_url('from'), 'local-target'])
    self.assertThat(self.hpss_calls, ContainsNoVfsCalls)
    self.assertLength(12, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)