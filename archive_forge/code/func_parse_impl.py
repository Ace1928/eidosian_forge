import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
@classmethod
def parse_impl(cls, buf, version=_DEFAULT_VERSION, from_zebra=False):
    if not from_zebra:
        prefix_len, = struct.unpack_from(cls._PREFIX_LEN_FMT, buf)
        rest = buf[cls.PREFIX_LEN_SIZE:]
        prefix = cls.PREFIX_CLS.bin_to_text(rest[:cls.PREFIX_LEN])
        return cls('%s/%d' % (prefix, prefix_len), from_zebra=False)
    prefix = cls.PREFIX_CLS.bin_to_text(buf[:cls.PREFIX_LEN])
    rest = buf[4:]
    metric, = struct.unpack_from(cls._METRIC_FMT, rest)
    rest = rest[cls.METRIC_SIZE:]
    nexthops, rest = _parse_nexthops(rest, version)
    return cls(prefix, metric, nexthops, from_zebra=True)