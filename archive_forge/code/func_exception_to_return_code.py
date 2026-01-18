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
def exception_to_return_code(the_callable, *args, **kwargs):
    """UI level helper for profiling and coverage.

    This transforms exceptions into a return value of 3. As such its only
    relevant to the UI layer, and should never be called where catching
    exceptions may be desirable.
    """
    try:
        return the_callable(*args, **kwargs)
    except (KeyboardInterrupt, Exception):
        exc_info = sys.exc_info()
        exitcode = trace.report_exception(exc_info, sys.stderr)
        if os.environ.get('BRZ_PDB'):
            print('**** entering debugger')
            import pdb
            pdb.post_mortem(exc_info[2])
        return exitcode