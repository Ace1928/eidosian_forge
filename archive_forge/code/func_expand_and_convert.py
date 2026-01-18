import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def expand_and_convert(val):
    if val is not None:
        if expand:
            if isinstance(val, str):
                val = self._expand_options_in_string(val)
            else:
                trace.warning('Cannot expand "%s": %s does not support option expansion' % (name, type(val)))
        if opt is None:
            val = found_store.unquote(val)
        elif convert:
            val = opt.convert_from_unicode(found_store, val)
    return val