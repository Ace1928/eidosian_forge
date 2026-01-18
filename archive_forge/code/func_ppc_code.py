import struct
from typing import Union
def ppc_code(self) -> int:
    limit: int = len(self.buffer) - 4
    i: int = 0
    while i <= limit:
        distance: int = self.current_position + i
        if self.buffer[i] & 252 == 72 and self.buffer[i + 3] & 3 == 1:
            src = struct.unpack('>L', self.buffer[i:i + 4])[0] & 67108860
            if self.is_encoder:
                dest = src + distance
            else:
                dest = src - distance
            dest = 72 << 24 | dest & 67108863 | 1
            self.buffer[i:i + 4] = struct.pack('>L', dest)
        i += 4
    self.current_position = i
    return i