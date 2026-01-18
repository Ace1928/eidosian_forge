from collections import namedtuple
import warnings
def setsampwidth(self, sampwidth):
    if self._nframeswritten:
        raise Error('cannot change parameters after starting to write')
    if sampwidth not in (1, 2, 3, 4):
        raise Error('bad sample width')
    self._sampwidth = sampwidth