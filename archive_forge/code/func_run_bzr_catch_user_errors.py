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
def run_bzr_catch_user_errors(argv):
    """Run brz and report user errors, but let internal errors propagate.

    This is used for the test suite, and might be useful for other programs
    that want to wrap the commandline interface.
    """
    install_bzr_command_hooks()
    try:
        return run_bzr(argv)
    except Exception as e:
        if isinstance(e, (OSError, IOError)) or not getattr(e, 'internal_error', True):
            trace.report_exception(sys.exc_info(), sys.stderr)
            return 3
        else:
            raise