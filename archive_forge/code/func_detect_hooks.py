from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def detect_hooks():
    """
    Returns True if the import hooks are installed, False if not.
    """
    flog.debug('Detecting hooks ...')
    present = any([hasattr(hook, 'RENAMER') for hook in sys.meta_path])
    if present:
        flog.debug('Detected.')
    else:
        flog.debug('Not detected.')
    return present