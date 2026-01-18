from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def scrub_py2_sys_modules():
    """
    Removes any Python 2 standard library modules from ``sys.modules`` that
    would interfere with Py3-style imports using import hooks. Examples are
    modules with the same names (like urllib or email).

    (Note that currently import hooks are disabled for modules like these
    with ambiguous names anyway ...)
    """
    if PY3:
        return {}
    scrubbed = {}
    for modulename in REPLACED_MODULES & set(RENAMES.keys()):
        if not modulename in sys.modules:
            continue
        module = sys.modules[modulename]
        if is_py2_stdlib_module(module):
            flog.debug('Deleting (Py2) {} from sys.modules'.format(modulename))
            scrubbed[modulename] = sys.modules[modulename]
            del sys.modules[modulename]
    return scrubbed