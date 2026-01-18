from taskflow import exceptions as exc
def check_flow_transition(old_state, new_state):
    """Check that flow can transition from ``old_state`` to ``new_state``.

    If transition can be performed, it returns true. If transition
    should be ignored, it returns false. If transition is not
    valid, it raises an InvalidState exception.
    """
    if old_state == new_state:
        return False
    pair = (old_state, new_state)
    if pair in _ALLOWED_FLOW_TRANSITIONS:
        return True
    if pair in _IGNORED_FLOW_TRANSITIONS:
        return False
    raise exc.InvalidState("Flow transition from '%s' to '%s' is not allowed" % pair)