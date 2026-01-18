from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_invalid_job_states(self):
    invalids = [(states.COMPLETE, states.UNCLAIMED), (states.UNCLAIMED, states.COMPLETE)]
    for start_state, end_state in invalids:
        self.assertRaises(excp.InvalidState, states.check_job_transition, start_state, end_state)