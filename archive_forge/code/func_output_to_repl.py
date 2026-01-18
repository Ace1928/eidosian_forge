import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
@contextmanager
def output_to_repl(repl):
    old_out, old_err = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = (repl.stdout, repl.stderr)
        yield
    finally:
        sys.stdout, sys.stderr = (old_out, old_err)