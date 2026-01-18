from pprint import pformat
from six import iteritems
import re
@burst.setter
def burst(self, burst):
    """
        Sets the burst of this V1alpha1WebhookThrottleConfig.
        ThrottleBurst is the maximum number of events sent at the same moment
        default 15 QPS

        :param burst: The burst of this V1alpha1WebhookThrottleConfig.
        :type: int
        """
    self._burst = burst