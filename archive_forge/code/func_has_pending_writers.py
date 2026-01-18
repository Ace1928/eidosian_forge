import collections
import contextlib
import threading
from fasteners import _utils
import six
@property
def has_pending_writers(self):
    """Returns if there are writers waiting to become the *one* writer."""
    return bool(self._pending_writers)