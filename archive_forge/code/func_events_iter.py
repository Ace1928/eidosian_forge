import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
def events_iter(self):
    """Returns iterator of events that can be registered/subscribed to.

        NOTE(harlowja): does not include back the ``ANY`` event type as that
        meta-type is not a specific event but is a capture-all that does not
        imply the same meaning as specific event types.
        """
    for event_type in self._watchable_events:
        yield event_type