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
def test_autowrap_store_files():
    x, y = symbols('x y')
    tmp = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(tmp)
    f = autowrap(x + y, backend='dummy', tempdir=tmp)
    assert f() == str(x + y)
    assert os.access(tmp, os.F_OK)
    TmpFileManager.cleanup()