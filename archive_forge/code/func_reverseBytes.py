from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def reverseBytes(s: bytes) -> bytes:
    fmt = str(len(s) // 2) + 'H'
    return struct.pack('<' + fmt, *struct.unpack('>' + fmt, s))