import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def iter_byte_chars(s):
    assert isinstance(s, bytes)
    return s