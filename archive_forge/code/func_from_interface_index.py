import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@classmethod
def from_interface_index(cls, context, ifindex):
    """
        Locate a device based on the interface index.

        :param `Context` context: the libudev context
        :param int ifindex: the interface index
        :returns: the device corresponding to the interface index
        :rtype: `Device`

        This method is only appropriate for network devices.
        """
    network_devices = context.list_devices(subsystem='net')
    dev = next((d for d in network_devices if d.attributes.get('ifindex') == ifindex), None)
    if dev is not None:
        return dev
    else:
        raise DeviceNotFoundByInterfaceIndexError(ifindex)