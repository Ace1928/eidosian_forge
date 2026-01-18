def is_cancellable_event(event):
    """Return if an event is cancellable by definition"""
    return event.startswith(BEFORE) or event.startswith(PRECOMMIT)