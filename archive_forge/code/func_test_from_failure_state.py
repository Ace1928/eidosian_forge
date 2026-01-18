from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_failure_state(self):
    self.assertTransitions(from_state=states.FAILURE, allowed=(states.REVERTING,), ignored=(states.FAILURE, states.RUNNING, states.PENDING, states.SUCCESS, states.REVERTED))