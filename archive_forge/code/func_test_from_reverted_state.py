from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_reverted_state(self):
    self.assertTransitions(from_state=states.REVERTED, allowed=(states.PENDING,), ignored=(states.REVERTING, states.REVERTED, states.RUNNING, states.SUCCESS, states.FAILURE))