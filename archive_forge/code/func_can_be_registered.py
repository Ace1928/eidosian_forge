import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
def can_be_registered(self, event_type):
    """Checks if the event can be registered/subscribed to.

        :param event_type: event that needs to be verified
        :returns: whether the event can be registered/subscribed to
        :rtype: boolean
        """
    return event_type in self._watchable_events or (event_type == self.ANY and self._allow_any)