from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_success_state(self):
    self.assertTransitions(from_state=states.SUCCESS, allowed=(states.REVERTING, states.RETRYING), ignored=(states.RUNNING, states.SUCCESS, states.PENDING, states.FAILURE, states.REVERTED))