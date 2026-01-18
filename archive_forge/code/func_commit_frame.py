from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def commit_frame(self, force=False):
    if self.current_frame:
        f = self.current_frame
        if f.tell() >= self._FRAME_SIZE_TARGET or force:
            data = f.getbuffer()
            write = self.file_write
            if len(data) >= self._FRAME_SIZE_MIN:
                write(FRAME + pack('<Q', len(data)))
            write(data)
            self.current_frame = io.BytesIO()