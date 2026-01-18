import inspect
def peek_objs(self):
    """Return iterator over stored objects ordered newest to oldest."""
    return (t_obj.obj for t_obj in reversed(self._stack))