import struct
from typing import Union
def armt_code(self) -> int:
    limit: int = len(self.buffer) - 4
    i: int = 0
    while i <= limit:
        if self.buffer[i + 1] & 248 == 240 and self.buffer[i + 3] & 248 == 248:
            src = self._unpack_thumb(self.buffer[i:i + 4]) << 1
            distance: int = self.current_position + i + 4
            if self.is_encoder:
                dest = src + distance
            else:
                dest = src - distance
            dest >>= 1
            self.buffer[i:i + 4] = self._pack_thumb(dest)
            i += 2
        i += 2
    self.current_position += i
    return i