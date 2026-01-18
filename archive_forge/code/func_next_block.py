import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def next_block(p):
    try:
        return next(block_iter[p])
    except StopIteration:
        return None