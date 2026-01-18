from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_from_reverting_state(self):
    self.assertTransitions(from_state=states.REVERTING, allowed=(states.REVERT_FAILURE, states.REVERTED), ignored=(states.RUNNING, states.REVERTING, states.PENDING, states.SUCCESS))