from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_invalid_task_states(self):
    invalids = [(states.RUNNING, states.PENDING), (states.PENDING, states.REVERTED), (states.PENDING, states.SUCCESS), (states.PENDING, states.FAILURE), (states.RETRYING, states.PENDING)]
    for start_state, end_state in invalids:
        self.assertFalse(states.check_task_transition(start_state, end_state))