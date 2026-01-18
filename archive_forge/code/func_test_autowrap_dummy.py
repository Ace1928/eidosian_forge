import os
import tempfile
import shutil
from io import StringIO
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np
def test_autowrap_dummy():
    x, y, z = symbols('x y z')
    f = autowrap(x + y, backend='dummy')
    assert f() == str(x + y)
    assert f.args == 'x, y'
    assert f.returns == 'nameless'
    f = autowrap(Eq(z, x + y), backend='dummy')
    assert f() == str(x + y)
    assert f.args == 'x, y'
    assert f.returns == 'z'
    f = autowrap(Eq(z, x + y + z), backend='dummy')
    assert f() == str(x + y + z)
    assert f.args == 'x, y, z'
    assert f.returns == 'z'