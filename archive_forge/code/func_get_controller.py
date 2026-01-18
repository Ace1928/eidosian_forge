import threading
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
@tf_contextlib.contextmanager
def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    self.stack.append(default)
    try:
        yield default
    finally:
        if self.stack:
            if self._enforce_nesting:
                if self.stack[-1] is not default:
                    raise AssertionError('Nesting violated for default stack of %s objects' % type(default))
                self.stack.pop()
            else:
                self.stack.remove(default)