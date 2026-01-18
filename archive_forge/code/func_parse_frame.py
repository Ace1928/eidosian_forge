import asyncio
import functools
import json
import random
import re
import sys
import zlib
from enum import IntEnum
from struct import Struct
from typing import (
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .helpers import NO_EXTENSIONS
from .streams import DataQueue
def parse_frame(self, buf: bytes) -> List[Tuple[bool, Optional[int], bytearray, Optional[bool]]]:
    """Return the next frame from the socket."""
    frames = []
    if self._tail:
        buf, self._tail = (self._tail + buf, b'')
    start_pos = 0
    buf_length = len(buf)
    while True:
        if self._state == WSParserState.READ_HEADER:
            if buf_length - start_pos >= 2:
                data = buf[start_pos:start_pos + 2]
                start_pos += 2
                first_byte, second_byte = data
                fin = first_byte >> 7 & 1
                rsv1 = first_byte >> 6 & 1
                rsv2 = first_byte >> 5 & 1
                rsv3 = first_byte >> 4 & 1
                opcode = first_byte & 15
                if rsv2 or rsv3 or (rsv1 and (not self._compress)):
                    raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received frame with non-zero reserved bits')
                if opcode > 7 and fin == 0:
                    raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received fragmented control frame')
                has_mask = second_byte >> 7 & 1
                length = second_byte & 127
                if opcode > 7 and length > 125:
                    raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Control frame payload cannot be larger than 125 bytes')
                if self._frame_fin or self._compressed is None:
                    self._compressed = True if rsv1 else False
                elif rsv1:
                    raise WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Received frame with non-zero reserved bits')
                self._frame_fin = bool(fin)
                self._frame_opcode = opcode
                self._has_mask = bool(has_mask)
                self._payload_length_flag = length
                self._state = WSParserState.READ_PAYLOAD_LENGTH
            else:
                break
        if self._state == WSParserState.READ_PAYLOAD_LENGTH:
            length = self._payload_length_flag
            if length == 126:
                if buf_length - start_pos >= 2:
                    data = buf[start_pos:start_pos + 2]
                    start_pos += 2
                    length = UNPACK_LEN2(data)[0]
                    self._payload_length = length
                    self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
                else:
                    break
            elif length > 126:
                if buf_length - start_pos >= 8:
                    data = buf[start_pos:start_pos + 8]
                    start_pos += 8
                    length = UNPACK_LEN3(data)[0]
                    self._payload_length = length
                    self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
                else:
                    break
            else:
                self._payload_length = length
                self._state = WSParserState.READ_PAYLOAD_MASK if self._has_mask else WSParserState.READ_PAYLOAD
        if self._state == WSParserState.READ_PAYLOAD_MASK:
            if buf_length - start_pos >= 4:
                self._frame_mask = buf[start_pos:start_pos + 4]
                start_pos += 4
                self._state = WSParserState.READ_PAYLOAD
            else:
                break
        if self._state == WSParserState.READ_PAYLOAD:
            length = self._payload_length
            payload = self._frame_payload
            chunk_len = buf_length - start_pos
            if length >= chunk_len:
                self._payload_length = length - chunk_len
                payload.extend(buf[start_pos:])
                start_pos = buf_length
            else:
                self._payload_length = 0
                payload.extend(buf[start_pos:start_pos + length])
                start_pos = start_pos + length
            if self._payload_length == 0:
                if self._has_mask:
                    assert self._frame_mask is not None
                    _websocket_mask(self._frame_mask, payload)
                frames.append((self._frame_fin, self._frame_opcode, payload, self._compressed))
                self._frame_payload = bytearray()
                self._state = WSParserState.READ_HEADER
            else:
                break
    self._tail = buf[start_pos:]
    return frames