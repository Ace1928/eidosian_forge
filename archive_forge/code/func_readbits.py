import logging
from io import BytesIO
from typing import BinaryIO, Iterator, List, Optional, cast
def readbits(self, bits: int) -> int:
    v = 0
    while 1:
        r = 8 - self.bpos
        if bits <= r:
            v = v << bits | self.buff >> r - bits & (1 << bits) - 1
            self.bpos += bits
            break
        else:
            v = v << r | self.buff & (1 << r) - 1
            bits -= r
            x = self.fp.read(1)
            if not x:
                raise EOFError
            self.buff = ord(x)
            self.bpos = 0
    return v