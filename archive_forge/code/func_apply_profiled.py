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
def apply_profiled(the_callable, *args, **kwargs):
    import tempfile
    import hotshot
    import hotshot.stats
    pffileno, pfname = tempfile.mkstemp()
    try:
        prof = hotshot.Profile(pfname)
        try:
            ret = prof.runcall(exception_to_return_code, the_callable, *args, **kwargs) or 0
        finally:
            prof.close()
        stats = hotshot.stats.load(pfname)
        stats.strip_dirs()
        stats.sort_stats('cum')
        stats.print_stats(20)
        return ret
    finally:
        os.close(pffileno)
        os.remove(pfname)