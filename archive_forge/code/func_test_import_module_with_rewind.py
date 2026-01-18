import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
def test_import_module_with_rewind(self):
    with self.tempfile() as (fullpath, path, modname):
        print(modname)
        with self.open(fullpath) as f:
            f.write('a = 0\n')
        self.head(path)
        self.push('import %s' % modname)
        self.assertIn(modname, self.repl.interp.locals)
        self.repl.undo()
        self.assertNotIn(modname, self.repl.interp.locals)
        self.repl.clear_modules_and_reevaluate()
        self.assertNotIn(modname, self.repl.interp.locals)
        self.push('import %s' % modname)
        self.push('a = %s.a' % modname)
        self.assertIn('a', self.repl.interp.locals)
        self.assertEqual(self.repl.interp.locals['a'], 0)
        with self.open(fullpath) as f:
            f.write('a = 1\n')
        self.repl.clear_modules_and_reevaluate()
        self.assertIn('a', self.repl.interp.locals)
        self.assertEqual(self.repl.interp.locals['a'], 1)