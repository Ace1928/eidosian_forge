import os
import shutil
import locket
import string
from toolz import memoize
from contextlib import contextmanager
from .utils import nested_get, flatten
def escape_filename(fn):
    """ Escape text so that it is a valid filename

    >>> escape_filename('Foo!bar?')
    'Foobar'

    """
    return ''.join(filter(valid_chars.__contains__, fn))