from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def format_position(position):
    if position:
        return u'%s:%d:%d: ' % (position[0].get_error_description(), position[1], position[2])
    return u''