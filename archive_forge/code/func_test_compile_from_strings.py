import sys
import os
import uuid
from importlib import import_module
import pytest
import numpy.f2py
from . import util
@pytest.mark.parametrize('fsource', ['program test_f2py\nend program test_f2py', b'program test_f2py\nend program test_f2py'])
def test_compile_from_strings(tmpdir, fsource):
    with util.switchdir(tmpdir):
        ret_val = numpy.f2py.compile(fsource, modulename='test_compile_from_strings', extension='.f90')
        assert ret_val == 0