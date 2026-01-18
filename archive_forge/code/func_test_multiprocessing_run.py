import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
def test_multiprocessing_run():
    """Set we can run mutiprocesgin without messing up up main namespace

    Note that import `nose.tools as nt` mdify the value s
    sys.module['__mp_main__'] so we need to temporarily set it to None to test
    the issue.
    """
    with TemporaryDirectory() as td:
        mpm = sys.modules.get('__mp_main__')
        sys.modules['__mp_main__'] = None
        try:
            path = pjoin(td, 'test.py')
            with open(path, 'w', encoding='utf-8') as f:
                f.write("import multiprocessing\nprint('hoy')")
            with capture_output() as io:
                _ip.run_line_magic('run', path)
                _ip.run_cell('i_m_undefined')
            out = io.stdout
            assert 'hoy' in out
            assert 'AttributeError' not in out
            assert 'NameError' in out
            assert out.count('---->') == 1
        except:
            raise
        finally:
            sys.modules['__mp_main__'] = mpm