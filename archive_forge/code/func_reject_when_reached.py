import futurist
def reject_when_reached(max_backlog):
    """Returns a function that will raise when backlog goes past max size."""

    def _rejector(executor, backlog):
        if backlog >= max_backlog:
            raise futurist.RejectedSubmission('Current backlog %s is not allowed to go beyond %s' % (backlog, max_backlog))
    return _rejector