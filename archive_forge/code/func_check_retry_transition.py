from taskflow import exceptions as exc
def check_retry_transition(old_state, new_state):
    """Check that retry can transition from ``old_state`` to ``new_state``.

    If transition can be performed, it returns true, false otherwise.
    """
    pair = (old_state, new_state)
    if pair in _ALLOWED_RETRY_TRANSITIONS:
        return True
    return False