from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def assertTransitionIgnored(self, from_state, to_state):
    msg = self._NOT_IGNORED_TPL % (from_state, to_state)
    self.assertFalse(self.check_transition(from_state, to_state), msg=msg)