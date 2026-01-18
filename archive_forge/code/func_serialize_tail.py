from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
def serialize_tail(self):
    return reduce(lambda a, b: a + b, (hdr.serialize() for hdr in self.lsa_headers))