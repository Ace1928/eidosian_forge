import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def inbound_body(self, buf):
    chunks = self._pending_chunks
    self.body_received += len(buf)
    if self.body_received >= self.body_size:
        if chunks:
            chunks.append(buf)
            self.body = bytes().join(chunks)
            chunks[:] = []
        else:
            self.body = buf
        self.ready = True
    else:
        chunks.append(buf)