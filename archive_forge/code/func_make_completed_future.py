import futurist
def make_completed_future(result):
    """Make and return a future completed with a given result."""
    future = futurist.Future()
    future.set_result(result)
    return future