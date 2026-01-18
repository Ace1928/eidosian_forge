import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def inbound_header(self, buf, offset=0):
    class_id, self.body_size = unpack_from('>HxxQ', buf, offset)
    offset += 12
    self._load_properties(class_id, buf, offset)
    if not self.body_size:
        self.ready = True
    return offset