import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@hi_resp_type.setter
def hi_resp_type(self, value: int) -> None:
    self._data[1:2] = struct.pack('B', value)