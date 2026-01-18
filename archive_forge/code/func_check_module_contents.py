import os
import platform
import pytest
import sys
import tempfile
import textwrap
import shutil
import random
import time
import traceback
from io import StringIO
from dataclasses import dataclass
import IPython.testing.tools as tt
from unittest import TestCase
from IPython.extensions.autoreload import AutoreloadMagics
from IPython.core.events import EventManager, pre_run_cell
from IPython.testing.decorators import skipif_not_numpy
from IPython.core.interactiveshell import ExecutionInfo
def check_module_contents():
    self.assertEqual(mod.x, 10)
    self.assertFalse(hasattr(mod, 'z'))
    self.assertEqual(old_foo(0), 4)
    self.assertEqual(mod.foo(0), 4)
    obj = mod.Baz(9)
    self.assertEqual(old_obj.bar(1), 11)
    self.assertEqual(obj.bar(1), 11)
    self.assertEqual(old_obj.quux, 43)
    self.assertEqual(obj.quux, 43)
    self.assertFalse(hasattr(old_obj, 'zzz'))
    self.assertFalse(hasattr(obj, 'zzz'))
    obj2 = mod.Bar()
    self.assertEqual(old_obj2.foo(), 2)
    self.assertEqual(obj2.foo(), 2)