import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
@property
def evbits(self):
    """
        Returns a dict with all supported event types and event codes, in
        the form of::

            {
              libevdev.EV_ABS: [libevdev.EV_ABS.ABS_X, ...],
              libevdev.EV_KEY: [libevdev.EV_KEY.BTN_LEFT, ...],
            }
        """
    types = {}
    for t in libevdev.types:
        if not self.has(t):
            continue
        codes = []
        for c in t.codes:
            if not self.has(c):
                continue
            codes.append(c)
        types[t] = codes
    return types