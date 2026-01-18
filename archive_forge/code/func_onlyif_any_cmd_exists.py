import sys
import os
import tempfile
import unittest
from ..py3compat import string_types, which
def onlyif_any_cmd_exists(*commands):
    """
    Decorator to skip test unless at least one of `commands` is found.
    """
    for cmd in commands:
        if which(cmd):
            return null_deco
    return skip('This test runs only if one of the commands {0} is installed'.format(commands))