import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@challenge_from_client.setter
def challenge_from_client(self, value: bytes) -> None:
    if len(value) != 8:
        raise ValueError('NTClientChallengeV2 ChallengeFromClient must be 8 bytes long')
    self._data[16:24] = value