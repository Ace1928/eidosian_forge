import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def next_read_size(self):
    if self.state_accept == self._state_accept_reading_unused:
        return 0
    elif self.decoding_failed:
        return 0
    elif self._number_needed_bytes is not None:
        return self._number_needed_bytes - self._in_buffer_len
    else:
        raise AssertionError("don't know how many bytes are expected!")