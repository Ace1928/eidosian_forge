from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def bytes_part_received(self, bytes):
    self._body_started = True
    self._bytes_parts.append(bytes)