import zlib
from typing import Optional, Tuple, Union
from .frame_protocol import CloseReason, FrameDecoder, FrameProtocol, Opcode, RsvBits
@client_max_window_bits.setter
def client_max_window_bits(self, value: int) -> None:
    if value < 9 or value > 15:
        raise ValueError('Window size must be between 9 and 15 inclusive')
    self._client_max_window_bits = value