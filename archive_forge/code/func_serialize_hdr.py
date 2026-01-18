import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
def serialize_hdr(self):
    """
        Serialization function for common part of authentication section.
        """
    return struct.pack(self._PACK_HDR_STR, self.auth_type, self.auth_len)