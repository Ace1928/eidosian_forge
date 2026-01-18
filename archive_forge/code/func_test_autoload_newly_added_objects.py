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
def test_autoload_newly_added_objects(self):
    self.shell.magic_autoreload('3')
    mod_code = '\n        def func1(): pass\n        '
    mod_name, mod_fn = self.new_module(textwrap.dedent(mod_code))
    self.shell.run_code(f'from {mod_name} import *')
    self.shell.run_code('func1()')
    with self.assertRaises(NameError):
        self.shell.run_code('func2()')
    with self.assertRaises(NameError):
        self.shell.run_code('t = Test()')
    with self.assertRaises(NameError):
        self.shell.run_code('number')
    new_code = "\n        def func1(): pass\n        def func2(): pass\n        class Test: pass\n        number = 0\n        from enum import Enum\n        class TestEnum(Enum):\n            A = 'a'\n        "
    self.write_file(mod_fn, textwrap.dedent(new_code))
    self.shell.run_code('func2()')
    self.shell.run_code(f"import sys; sys.modules['{mod_name}'].func2()")
    self.shell.run_code('t = Test()')
    self.shell.run_code('number')
    self.shell.run_code('TestEnum.A')
    new_code = "\n        def func1(): return 'changed'\n        def func2(): return 'changed'\n        class Test:\n            def new_func(self):\n                return 'changed'\n        number = 1\n        from enum import Enum\n        class TestEnum(Enum):\n            A = 'a'\n            B = 'added'\n        "
    self.write_file(mod_fn, textwrap.dedent(new_code))
    self.shell.run_code("assert func1() == 'changed'")
    self.shell.run_code("assert func2() == 'changed'")
    self.shell.run_code("t = Test(); assert t.new_func() == 'changed'")
    self.shell.run_code('assert number == 1')
    if sys.version_info < (3, 12):
        self.shell.run_code("assert TestEnum.B.value == 'added'")
    new_mod_code = "\n        from enum import Enum\n        class Ext(Enum):\n            A = 'ext'\n        def ext_func():\n            return 'ext'\n        class ExtTest:\n            def meth(self):\n                return 'ext'\n        ext_int = 2\n        "
    new_mod_name, new_mod_fn = self.new_module(textwrap.dedent(new_mod_code))
    current_mod_code = f'\n        from {new_mod_name} import *\n        '
    self.write_file(mod_fn, textwrap.dedent(current_mod_code))
    self.shell.run_code("assert Ext.A.value == 'ext'")
    self.shell.run_code("assert ext_func() == 'ext'")
    self.shell.run_code("t = ExtTest(); assert t.meth() == 'ext'")
    self.shell.run_code('assert ext_int == 2')