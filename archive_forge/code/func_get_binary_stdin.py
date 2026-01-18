import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def get_binary_stdin() -> t.BinaryIO:
    reader = _find_binary_reader(sys.stdin)
    if reader is None:
        raise RuntimeError('Was not able to determine binary stream for sys.stdin.')
    return reader