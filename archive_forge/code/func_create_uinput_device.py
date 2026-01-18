import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
def create_uinput_device(self, uinput_fd=None):
    """
        Creates and returns a new :class:`Device` based on this libevdev
        device. The new device is equivalent to one created with
        ``libevdev.Device()``, i.e. it is one that does not have a file
        descriptor associated.

        To create a uinput device from an existing device::

            fd = open('/dev/input/event0', 'rb')
            d = libevdev.Device(fd)
            d.name = 'duplicated device'
            d.create_uinput_device()
            # d is now a duplicate of the event0 device with a custom name
            fd.close()

        Or to create a new device from scratch::

            d = libevdev.Device()
            d.name = 'test device'
            d.enable(libevdev.EV_KEY.BTN_LEFT)
            d.create_uinput_device()
            # d is now a device with a single button

        :param uinput_fd: A file descriptor to the /dev/input/uinput device. If None, the device is opened and closed automatically.
        :raises: OSError
        """
    d = libevdev.Device()
    d.name = self.name
    d.id = self.id
    for t, cs in self.evbits.items():
        for c in cs:
            if t == libevdev.EV_ABS:
                data = self.absinfo[c]
            elif t == libevdev.EV_REP:
                data = self.value[c]
            else:
                data = None
            d.enable(c, data)
    for p in self.properties:
        self.enable(p)
    d._uinput = UinputDevice(self._libevdev, uinput_fd)
    return d