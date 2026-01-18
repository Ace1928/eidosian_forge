import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@custom_data.setter
def custom_data(self, value: bytes) -> None:
    if len(value) != 8:
        raise ValueError('custom_data length must be 8 bytes long')
    self._data[8:16] = value