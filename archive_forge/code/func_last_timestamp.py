from pprint import pformat
from six import iteritems
import re
@last_timestamp.setter
def last_timestamp(self, last_timestamp):
    """
        Sets the last_timestamp of this V1Event.
        The time at which the most recent occurrence of this event was recorded.

        :param last_timestamp: The last_timestamp of this V1Event.
        :type: datetime
        """
    self._last_timestamp = last_timestamp