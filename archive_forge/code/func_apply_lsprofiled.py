import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def apply_lsprofiled(filename, the_callable, *args, **kwargs):
    from breezy.lsprof import profile
    ret, stats = profile(exception_to_return_code, the_callable, *args, **kwargs)
    stats.sort()
    if filename is None:
        stats.pprint()
    else:
        stats.save(filename)
        trace.note(gettext('Profile data written to "%s".'), filename)
    return ret