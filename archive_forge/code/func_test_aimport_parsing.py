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
def test_aimport_parsing(self):
    module_reloader = self.shell.auto_magics._reloader
    self.shell.magic_aimport('os')
    assert module_reloader.modules['os'] is True
    assert 'os' not in module_reloader.skip_modules.keys()
    self.shell.magic_aimport('-math')
    assert module_reloader.skip_modules['math'] is True
    assert 'math' not in module_reloader.modules.keys()
    self.shell.magic_aimport('-os, math')
    assert module_reloader.modules['math'] is True
    assert 'math' not in module_reloader.skip_modules.keys()
    assert module_reloader.skip_modules['os'] is True
    assert 'os' not in module_reloader.modules.keys()