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
def parse_pre(cls, buf):
    buf = bytes(buf)
    header_fields, _ = cls.parse_common_header(buf)
    type_ = header_fields[1]
    length = header_fields[3]
    if type_ in cls._EXT_TS_TYPES:
        header_cls = ExtendedTimestampMrtRecord
    else:
        header_cls = MrtCommonRecord
    required_len = header_cls.HEADER_SIZE + length
    return required_len