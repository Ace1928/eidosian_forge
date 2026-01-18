from pprint import pformat
from six import iteritems
import re
@deprecated_source.setter
def deprecated_source(self, deprecated_source):
    """
        Sets the deprecated_source of this V1beta1Event.
        Deprecated field assuring backward compatibility with core.v1 Event type

        :param deprecated_source: The deprecated_source of this V1beta1Event.
        :type: V1EventSource
        """
    self._deprecated_source = deprecated_source