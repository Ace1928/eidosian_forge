from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def assertTransitionForbidden(self, from_state, to_state):
    self.assertRaisesRegex(exc.InvalidState, self.transition_exc_regexp, self.check_transition, from_state, to_state)