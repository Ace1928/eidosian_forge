from pprint import pformat
from six import iteritems
import re
@retry_after_seconds.setter
def retry_after_seconds(self, retry_after_seconds):
    """
        Sets the retry_after_seconds of this V1StatusDetails.
        If specified, the time in seconds before the operation should be
        retried. Some errors may indicate the client must take an alternate
        action - for those errors this field may indicate how long to wait
        before taking the alternate action.

        :param retry_after_seconds: The retry_after_seconds of this
        V1StatusDetails.
        :type: int
        """
    self._retry_after_seconds = retry_after_seconds