import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
@classmethod
def get_link_dirs(cls, context):
    """
        Get all directories that may contain links to device nodes.

        This method checks the device links of every device, so it is very
        expensive.

        :param Context context: the context
        :returns: a sorted list of directories that contain device links
        :rtype: list
        """
    devices = context.list_devices()
    devices_with_links = (d for d in devices if list(d.device_links))
    links = (l for d in devices_with_links for l in d.device_links)
    return sorted(set((os.path.dirname(l) for l in links)))