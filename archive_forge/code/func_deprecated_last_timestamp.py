from pprint import pformat
from six import iteritems
import re
@deprecated_last_timestamp.setter
def deprecated_last_timestamp(self, deprecated_last_timestamp):
    """
        Sets the deprecated_last_timestamp of this V1beta1Event.
        Deprecated field assuring backward compatibility with core.v1 Event type

        :param deprecated_last_timestamp: The deprecated_last_timestamp of this
        V1beta1Event.
        :type: datetime
        """
    self._deprecated_last_timestamp = deprecated_last_timestamp