from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_ignored_job_states(self):
    ignored = []
    for start_state, end_state in states._ALLOWED_JOB_TRANSITIONS:
        ignored.append((start_state, start_state))
        ignored.append((end_state, end_state))
    for start_state, end_state in ignored:
        self.assertFalse(states.check_job_transition(start_state, end_state))