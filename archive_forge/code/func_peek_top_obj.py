import inspect
def peek_top_obj(self):
    """Return the most recent stored object."""
    return self._stack[-1].obj