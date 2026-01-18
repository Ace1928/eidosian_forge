import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
def check_parameters(self):
    assert self.root_priority % self._BRIDGE_PRIORITY_STEP == 0
    assert self.bridge_priority % self._BRIDGE_PRIORITY_STEP == 0
    assert self.port_priority % self._PORT_PRIORITY_STEP == 0
    assert self.message_age % self._TIMER_STEP == 0
    assert self.max_age % self._TIMER_STEP == 0
    assert self.hello_time % self._TIMER_STEP == 0
    assert self.forward_delay % self._TIMER_STEP == 0