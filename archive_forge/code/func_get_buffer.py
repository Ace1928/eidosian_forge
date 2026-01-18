import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def get_buffer(self, n):
    want = n
    if want <= 0 or want > self.max_size:
        want = self.max_size
    if len(self._ssl_buffer) < want:
        self._ssl_buffer = bytearray(want)
        self._ssl_buffer_view = memoryview(self._ssl_buffer)
    return self._ssl_buffer_view