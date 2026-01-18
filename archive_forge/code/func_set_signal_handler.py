import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def set_signal_handler(signum, handler, restart_syscall=True):
    """A wrapper for signal.signal that also calls siginterrupt(signum, False)
    on platforms that support that.

    :param restart_syscall: if set, allow syscalls interrupted by a signal to
        automatically restart (by calling `signal.siginterrupt(signum,
        False)`).  May be ignored if the feature is not available on this
        platform or Python version.
    """
    try:
        import signal
        siginterrupt = signal.siginterrupt
    except ImportError:
        return None
    except AttributeError:

        def siginterrupt(signum, flag):
            return None
    if restart_syscall:

        def sig_handler(*args):
            siginterrupt(signum, False)
            handler(*args)
    else:
        sig_handler = handler
    old_handler = signal.signal(signum, sig_handler)
    if restart_syscall:
        siginterrupt(signum, False)
    return old_handler