from pprint import pformat
from six import iteritems
import re
@default_request.setter
def default_request(self, default_request):
    """
        Sets the default_request of this V1LimitRangeItem.
        DefaultRequest is the default resource requirement request value by
        resource name if resource request is omitted.

        :param default_request: The default_request of this V1LimitRangeItem.
        :type: dict(str, str)
        """
    self._default_request = default_request