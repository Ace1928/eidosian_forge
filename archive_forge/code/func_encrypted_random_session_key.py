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
def encrypted_random_session_key(self) -> typing.Optional[bytes]:
    """The client's encrypted random session key."""
    return _unpack_payload(self._data, 52)