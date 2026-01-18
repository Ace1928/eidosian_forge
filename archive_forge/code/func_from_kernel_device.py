import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@classmethod
def from_kernel_device(cls, context, kernel_device):
    """
        Locate a device based on the kernel device.

        :param `Context` context: the libudev context
        :param str kernel_device: the kernel device
        :returns: the device corresponding to ``kernel_device``
        :rtype: `Device`
        """
    switch_char = kernel_device[0]
    rest = kernel_device[1:]
    if switch_char in ('b', 'c'):
        number_re = re.compile('^(?P<major>\\d+):(?P<minor>\\d+)$')
        match = number_re.match(rest)
        if match:
            number = os.makedev(int(match.group('major')), int(match.group('minor')))
            return cls.from_device_number(context, switch_char, number)
        else:
            raise DeviceNotFoundByKernelDeviceError(kernel_device)
    elif switch_char == 'n':
        return cls.from_interface_index(context, rest)
    elif switch_char == '+':
        subsystem, _, kernel_device_name = rest.partition(':')
        if kernel_device_name and subsystem:
            return cls.from_name(context, subsystem, kernel_device_name)
        else:
            raise DeviceNotFoundByKernelDeviceError(kernel_device)
    else:
        raise DeviceNotFoundByKernelDeviceError(kernel_device)