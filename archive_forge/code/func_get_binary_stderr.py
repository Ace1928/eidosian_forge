import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def get_binary_stderr() -> t.BinaryIO:
    writer = _find_binary_writer(sys.stderr)
    if writer is None:
        raise RuntimeError('Was not able to determine binary stream for sys.stderr.')
    return writer