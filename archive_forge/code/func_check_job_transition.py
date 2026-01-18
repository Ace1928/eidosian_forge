from taskflow import exceptions as exc
def check_job_transition(old_state, new_state):
    """Check that job can transition from from ``old_state`` to ``new_state``.

    If transition can be performed, it returns true. If transition
    should be ignored, it returns false. If transition is not
    valid, it raises an InvalidState exception.
    """
    if old_state == new_state:
        return False
    pair = (old_state, new_state)
    if pair in _ALLOWED_JOB_TRANSITIONS:
        return True
    raise exc.InvalidState("Job transition from '%s' to '%s' is not allowed" % pair)