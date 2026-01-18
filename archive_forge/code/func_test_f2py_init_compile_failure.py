import sys
import os
import uuid
from importlib import import_module
import pytest
import numpy.f2py
from . import util
def test_f2py_init_compile_failure():
    ret_val = numpy.f2py.compile(b'invalid')
    assert ret_val == 1