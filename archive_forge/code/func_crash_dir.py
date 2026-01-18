import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def crash_dir():
    """Return the directory name to store crash files.

    This doesn't implicitly create it.

    On Windows it's in the config directory; elsewhere it's /var/crash
    which may be monitored by apport.  It can be overridden by
    $APPORT_CRASH_DIR.
    """
    if sys.platform == 'win32':
        return osutils.pathjoin(config_dir(), 'Crash')
    else:
        return os.environ.get('APPORT_CRASH_DIR', '/var/crash')