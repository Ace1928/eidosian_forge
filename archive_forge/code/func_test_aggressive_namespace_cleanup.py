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
def test_aggressive_namespace_cleanup(self):
    """Test that namespace cleanup is not too aggressive GH-238

        Returning from another run magic deletes the namespace"""
    with tt.TempFileMixin() as empty:
        empty.mktmp('')
        src = 'ip = get_ipython()\nfor i in range(5):\n   try:\n       ip.magic(%r)\n   except NameError as e:\n       print(i)\n       break\n' % ('run ' + empty.fname)
        self.mktmp(src)
        _ip.run_line_magic('run', str(self.fname))
        _ip.run_cell('ip == get_ipython()')
        assert _ip.user_ns['i'] == 4