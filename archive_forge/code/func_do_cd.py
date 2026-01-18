import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def do_cd(self, test_case, input, args):
    if len(args) > 1:
        raise SyntaxError('Usage: cd [dir]')
    if len(args) == 1:
        d = args[0]
        self._ensure_in_jail(test_case, d)
    else:
        d = self._get_jail_root(test_case)
    os.chdir(d)
    return (0, None, None)