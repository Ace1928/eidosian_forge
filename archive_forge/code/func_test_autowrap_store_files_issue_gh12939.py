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
def test_autowrap_store_files_issue_gh12939():
    x, y = symbols('x y')
    tmp = './tmp'
    saved_cwd = os.getcwd()
    temp_cwd = tempfile.mkdtemp()
    try:
        os.chdir(temp_cwd)
        f = autowrap(x + y, backend='dummy', tempdir=tmp)
        assert f() == str(x + y)
        assert os.access(tmp, os.F_OK)
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(temp_cwd)