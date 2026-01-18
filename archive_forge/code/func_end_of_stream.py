import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def end_of_stream(self):
    b = self.mat_stream.read(1)
    curpos = self.mat_stream.tell()
    self.mat_stream.seek(curpos - 1)
    return len(b) == 0