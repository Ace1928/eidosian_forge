from pprint import pformat
from six import iteritems
import re
@ssl_enabled.setter
def ssl_enabled(self, ssl_enabled):
    """
        Sets the ssl_enabled of this V1ScaleIOVolumeSource.
        Flag to enable/disable SSL communication with Gateway, default false

        :param ssl_enabled: The ssl_enabled of this V1ScaleIOVolumeSource.
        :type: bool
        """
    self._ssl_enabled = ssl_enabled