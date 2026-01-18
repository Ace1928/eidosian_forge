from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_retrying_state(self):
    self.assertTransitions(from_state=states.RETRYING, allowed=(states.RUNNING,), ignored=(states.RETRYING, states.SUCCESS, states.PENDING, states.FAILURE, states.REVERTED))