from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def remove_hooks(scrub_sys_modules=False):
    """
    This function removes the import hook from sys.meta_path.
    """
    if PY3:
        return
    flog.debug('Uninstalling hooks ...')
    for i, hook in list(enumerate(sys.meta_path))[::-1]:
        if hasattr(hook, 'RENAMER'):
            del sys.meta_path[i]
    if scrub_sys_modules:
        scrub_future_sys_modules()