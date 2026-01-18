import struct
import builtins
import warnings
from collections import namedtuple
def aifc(self):
    if self._nframeswritten:
        raise Error('cannot change parameters after starting to write')
    self._aifc = 1