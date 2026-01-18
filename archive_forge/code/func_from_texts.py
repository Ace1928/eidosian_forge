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
@classmethod
def from_texts(cls, text, parents=()):
    """Produce a MultiParent from a text and list of parent text"""
    return cls.from_lines(BytesIO(text).readlines(), [BytesIO(p).readlines() for p in parents])