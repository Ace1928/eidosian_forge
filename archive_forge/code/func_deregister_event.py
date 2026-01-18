import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
def deregister_event(self, event_type):
    """Remove a group of listeners bound to event ``event_type``.

        :param event_type: deregister listeners bound to event_type
        """
    return len(self._topics.pop(event_type, []))