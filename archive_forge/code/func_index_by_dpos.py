from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def index_by_dpos(self, pos):
    if pos < 0:
        pos = 0
    i = bisect_right(self._cumulated_d_size, pos)
    if i != self._frames_count + 1:
        return i
    else:
        return None