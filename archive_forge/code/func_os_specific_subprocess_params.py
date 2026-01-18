import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def os_specific_subprocess_params():
    """Get O/S specific subprocess parameters."""
    if sys.platform == 'win32':
        return {}
    else:
        return {'preexec_fn': _ignore_signals, 'close_fds': True}