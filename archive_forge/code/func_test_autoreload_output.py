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
def test_autoreload_output(self):
    self.shell.magic_autoreload('complete')
    mod_code = '\n        def func1(): pass\n        '
    mod_name, mod_fn = self.new_module(mod_code)
    self.shell.run_code(f'import {mod_name}')
    with tt.AssertPrints('', channel='stdout'):
        self.shell.run_code('pass')
    self.shell.magic_autoreload('complete --print')
    self.write_file(mod_fn, mod_code)
    with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
        self.shell.run_code('pass')
    self.shell.magic_autoreload('complete -p')
    self.write_file(mod_fn, mod_code)
    with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
        self.shell.run_code('pass')
    self.shell.magic_autoreload('complete --print --log')
    self.write_file(mod_fn, mod_code)
    with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
        self.shell.run_code('pass')
    self.shell.magic_autoreload('complete --print --log')
    self.write_file(mod_fn, mod_code)
    with self.assertLogs(logger='autoreload') as lo:
        self.shell.run_code('pass')
    assert lo.output == [f"INFO:autoreload:Reloading '{mod_name}'."]
    self.shell.magic_autoreload('complete -l')
    self.write_file(mod_fn, mod_code)
    with self.assertLogs(logger='autoreload') as lo:
        self.shell.run_code('pass')
    assert lo.output == [f"INFO:autoreload:Reloading '{mod_name}'."]