import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
@classmethod
def native_labels(cls) -> typing.Dict['MessageType', str]:
    return {MessageType.negotiate: 'NEGOTIATE_MESSAGE', MessageType.challenge: 'CHALLENGE_MESSAGE', MessageType.authenticate: 'AUTHENTICATE_MESSAGE'}