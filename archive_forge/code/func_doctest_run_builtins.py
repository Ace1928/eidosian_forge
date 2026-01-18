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
def doctest_run_builtins():
    """Check that %run doesn't damage __builtins__.

    In [1]: import tempfile

    In [2]: bid1 = id(__builtins__)

    In [3]: fname = tempfile.mkstemp('.py')[1]

    In [3]: f = open(fname, 'w', encoding='utf-8')

    In [4]: dummy= f.write('pass\\n')

    In [5]: f.flush()

    In [6]: t1 = type(__builtins__)

    In [7]: %run $fname

    In [7]: f.close()

    In [8]: bid2 = id(__builtins__)

    In [9]: t2 = type(__builtins__)

    In [10]: t1 == t2
    Out[10]: True

    In [10]: bid1 == bid2
    Out[10]: True

    In [12]: try:
       ....:     os.unlink(fname)
       ....: except:
       ....:     pass
       ....:
    """