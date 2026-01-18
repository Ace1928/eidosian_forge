import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def ensure_string_list(self, option):
    """Ensure that 'option' is a list of strings.  If 'option' is
        currently a string, we split it either on /,\\s*/ or /\\s+/, so
        "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become
        ["foo", "bar", "baz"].
        """
    val = getattr(self, option)
    if val is None:
        return
    elif isinstance(val, str):
        setattr(self, option, re.split(',\\s*|\\s+', val))
    else:
        if isinstance(val, list):
            ok = all((isinstance(v, str) for v in val))
        else:
            ok = False
        if not ok:
            raise DistutilsOptionError("'{}' must be a list of strings (got {!r})".format(option, val))