import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
def test_macro_run():
    """Test that we can run a multi-line macro successfully."""
    ip = get_ipython()
    ip.history_manager.reset()
    cmds = ['a=10', 'a+=1', 'print(a)', '%macro test 2-3']
    for cmd in cmds:
        ip.run_cell(cmd, store_history=True)
    assert ip.user_ns['test'].value == 'a+=1\nprint(a)\n'
    with tt.AssertPrints('12'):
        ip.run_cell('test')
    with tt.AssertPrints('13'):
        ip.run_cell('test')