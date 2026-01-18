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
@classmethod
def parse_rib_entries(cls, buf):
    entry_count, = struct.unpack_from('!H', buf)
    rest = buf[2:]
    rib_entries = []
    for i in range(entry_count):
        r, rest = MrtRibEntry.parse(rest, is_addpath=cls._IS_ADDPATH)
        rib_entries.insert(i, r)
    return (entry_count, rib_entries, rest)