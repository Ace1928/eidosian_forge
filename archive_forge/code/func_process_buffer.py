import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def process_buffer(self) -> Optional[Frame]:
    if not self.header:
        if not self.parse_header():
            return None
    assert self.header is not None
    assert self.masker is not None
    assert self.effective_opcode is not None
    if len(self.buffer) < self.payload_required:
        return None
    payload_remaining = self.header.payload_len - self.payload_consumed
    payload = self.buffer.consume_at_most(payload_remaining)
    if not payload and self.header.payload_len > 0:
        return None
    self.buffer.commit()
    self.payload_consumed += len(payload)
    finished = self.payload_consumed == self.header.payload_len
    payload = self.masker.process(payload)
    for extension in self.extensions:
        payload_ = extension.frame_inbound_payload_data(self, payload)
        if isinstance(payload_, CloseReason):
            raise ParseFailed('error in extension', payload_)
        payload = payload_
    if finished:
        final = bytearray()
        for extension in self.extensions:
            result = extension.frame_inbound_complete(self, self.header.fin)
            if isinstance(result, CloseReason):
                raise ParseFailed('error in extension', result)
            if result is not None:
                final += result
        payload += final
    frame = Frame(self.effective_opcode, payload, finished, self.header.fin)
    if finished:
        self.header = None
        self.effective_opcode = None
        self.masker = None
    else:
        self.effective_opcode = Opcode.CONTINUATION
    return frame