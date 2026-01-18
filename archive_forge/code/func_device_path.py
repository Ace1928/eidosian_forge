from pprint import pformat
from six import iteritems
import re
@device_path.setter
def device_path(self, device_path):
    """
        Sets the device_path of this V1VolumeDevice.
        devicePath is the path inside of the container that the device will be
        mapped to.

        :param device_path: The device_path of this V1VolumeDevice.
        :type: str
        """
    if device_path is None:
        raise ValueError('Invalid value for `device_path`, must not be `None`')
    self._device_path = device_path