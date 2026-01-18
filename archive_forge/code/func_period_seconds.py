from pprint import pformat
from six import iteritems
import re
@period_seconds.setter
def period_seconds(self, period_seconds):
    """
        Sets the period_seconds of this V1Probe.
        How often (in seconds) to perform the probe. Default to 10 seconds.
        Minimum value is 1.

        :param period_seconds: The period_seconds of this V1Probe.
        :type: int
        """
    self._period_seconds = period_seconds