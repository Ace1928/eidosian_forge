from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
@property
def seek_table_info(self):
    """A tuple: (frames_number, compressed_size, decompressed_size)
        1, Frames_number and compressed_size don't count the seek table
           frame (a zstd skippable frame at the end of the file).
        2, In write modes, the part of data that has not been flushed to
           frames is not counted.
        3, If the SeekableZstdFile object is closed, it's None.
        """
    if self._mode == _MODE_WRITE:
        return self._seek_table.get_info()
    elif self._mode == _MODE_READ:
        return self._buffer.raw.get_seek_table_info()
    else:
        return None