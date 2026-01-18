import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@property
def reserved(self) -> bytes:
    return self._data[4:7].tobytes()