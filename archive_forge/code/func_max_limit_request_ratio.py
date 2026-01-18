from pprint import pformat
from six import iteritems
import re
@max_limit_request_ratio.setter
def max_limit_request_ratio(self, max_limit_request_ratio):
    """
        Sets the max_limit_request_ratio of this V1LimitRangeItem.
        MaxLimitRequestRatio if specified, the named resource must have a
        request and limit that are both non-zero where limit divided by request
        is less than or equal to the enumerated value; this represents the max
        burst for the named resource.

        :param max_limit_request_ratio: The max_limit_request_ratio of this
        V1LimitRangeItem.
        :type: dict(str, str)
        """
    self._max_limit_request_ratio = max_limit_request_ratio