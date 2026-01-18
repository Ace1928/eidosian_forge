import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
def listeners_iter(self):
    """Return an iterator over the mapping of event => listeners bound.

        NOTE(harlowja): Each listener in the yielded (event, listeners)
        tuple is an instance of the :py:class:`~.Listener`  type, which
        itself wraps a provided callback (and its details filter
        callback, if any).
        """
    for event_type, listeners in self._topics.items():
        if listeners:
            yield (event_type, listeners)