import inspect
def pop_obj(self):
    """Remove last-inserted object and return it, without filename/line info."""
    return self._stack.pop().obj