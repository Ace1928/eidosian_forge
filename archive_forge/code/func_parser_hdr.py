import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@classmethod
def parser_hdr(cls, buf):
    """
        Parser for common part of authentication section.
        """
    return struct.unpack_from(cls._PACK_HDR_STR, buf[:cls._PACK_HDR_STR_LEN])