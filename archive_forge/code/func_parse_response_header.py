import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
def parse_response_header(self, read_buffer):
    if self.FLEXIBLE_VERSION:
        return ResponseHeaderV2.decode(read_buffer)
    return ResponseHeader.decode(read_buffer)