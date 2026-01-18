from pprint import pformat
from six import iteritems
import re
@deprecated_count.setter
def deprecated_count(self, deprecated_count):
    """
        Sets the deprecated_count of this V1beta1Event.
        Deprecated field assuring backward compatibility with core.v1 Event type

        :param deprecated_count: The deprecated_count of this V1beta1Event.
        :type: int
        """
    self._deprecated_count = deprecated_count