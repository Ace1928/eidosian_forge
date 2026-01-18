import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
def serialize_rib_entries(self):
    self.entry_count = len(self.rib_entries)
    rib_entries_bin = bytearray()
    for r in self.rib_entries:
        rib_entries_bin += r.serialize()
    return struct.pack('!H', self.entry_count) + rib_entries_bin