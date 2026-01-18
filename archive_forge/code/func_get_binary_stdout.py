import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def get_binary_stdout() -> t.BinaryIO:
    writer = _find_binary_writer(sys.stdout)
    if writer is None:
        raise RuntimeError('Was not able to determine binary stream for sys.stdout.')
    return writer