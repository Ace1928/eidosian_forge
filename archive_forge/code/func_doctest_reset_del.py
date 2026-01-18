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
def doctest_reset_del():
    """Test that resetting doesn't cause errors in __del__ methods.

    In [2]: class A(object):
       ...:     def __del__(self):
       ...:         print(str("Hi"))
       ...:

    In [3]: a = A()

    In [4]: get_ipython().reset(); import gc; x = gc.collect(0)
    Hi

    In [5]: 1+1
    Out[5]: 2
    """