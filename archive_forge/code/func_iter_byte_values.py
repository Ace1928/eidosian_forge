import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def iter_byte_values(s):
    assert isinstance(s, bytes)
    return (ord(c) for c in s)