from pprint import pformat
from six import iteritems
import re
@last_probe_time.setter
def last_probe_time(self, last_probe_time):
    """
        Sets the last_probe_time of this V1PodCondition.
        Last time we probed the condition.

        :param last_probe_time: The last_probe_time of this V1PodCondition.
        :type: datetime
        """
    self._last_probe_time = last_probe_time