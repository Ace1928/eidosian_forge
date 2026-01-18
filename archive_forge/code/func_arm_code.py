import struct
from typing import Union
def arm_code(self) -> int:
    limit = len(self.buffer) - 4
    i = 0
    while i <= limit:
        if self.buffer[i + 3] == 235:
            src = struct.unpack('<L', self.buffer[i:i + 3] + b'\x00')[0] << 2
            distance = self.current_position + i + 8
            if self.is_encoder:
                dest = src + distance >> 2
            else:
                dest = src - distance >> 2
            self.buffer[i:i + 3] = struct.pack('<L', dest & 16777215)[:3]
        i += 4
    self.current_position += i
    return i