import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def pad_to_width(line, width, encoding_hint='ascii'):
    """Truncate or pad unicode line to width.

    This is best-effort for now, and strings containing control codes or
    non-ascii text may be cut and padded incorrectly.
    """
    s = line.encode(encoding_hint, 'replace')
    return (b'%-*.*s' % (width, width, s)).decode(encoding_hint)