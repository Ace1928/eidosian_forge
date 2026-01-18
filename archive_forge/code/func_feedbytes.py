import array
from typing import (
def feedbytes(self, data: bytes) -> None:
    for byte in get_bytes(data):
        try:
            for m in (128, 64, 32, 16, 8, 4, 2, 1):
                self._parse_bit(byte & m)
        except self.ByteSkip:
            self._accept = self._parse_mode
            self._state = self.MODE
        except self.EOFB:
            break
    return