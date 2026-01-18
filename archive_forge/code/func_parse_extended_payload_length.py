import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def parse_extended_payload_length(self, opcode: Opcode, payload_len: int) -> Optional[int]:
    if opcode.iscontrol() and payload_len > MAX_PAYLOAD_NORMAL:
        raise ParseFailed('Control frame with payload len > 125')
    if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
        data = self.buffer.consume_exactly(2)
        if data is None:
            return None
        payload_len, = struct.unpack('!H', data)
        if payload_len <= MAX_PAYLOAD_NORMAL:
            raise ParseFailed('Payload length used 2 bytes when 1 would have sufficed')
    elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
        data = self.buffer.consume_exactly(8)
        if data is None:
            return None
        payload_len, = struct.unpack('!Q', data)
        if payload_len <= MAX_PAYLOAD_TWO_BYTE:
            raise ParseFailed('Payload length used 8 bytes when 2 would have sufficed')
        if payload_len >> 63:
            raise ParseFailed('8-byte payload length with non-zero MSB')
    return payload_len