from pprint import pformat
from six import iteritems
import re
@deprecated_first_timestamp.setter
def deprecated_first_timestamp(self, deprecated_first_timestamp):
    """
        Sets the deprecated_first_timestamp of this V1beta1Event.
        Deprecated field assuring backward compatibility with core.v1 Event type

        :param deprecated_first_timestamp: The deprecated_first_timestamp of
        this V1beta1Event.
        :type: datetime
        """
    self._deprecated_first_timestamp = deprecated_first_timestamp