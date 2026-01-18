from pprint import pformat
from six import iteritems
import re
@failure_threshold.setter
def failure_threshold(self, failure_threshold):
    """
        Sets the failure_threshold of this V1Probe.
        Minimum consecutive failures for the probe to be considered failed after
        having succeeded. Defaults to 3. Minimum value is 1.

        :param failure_threshold: The failure_threshold of this V1Probe.
        :type: int
        """
    self._failure_threshold = failure_threshold