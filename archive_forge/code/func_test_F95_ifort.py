from sympy.abc import x, y, z
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess
def test_F95_ifort():
    if ('F95', 'ifort') in invalid_lang_compilers:
        skip("`ifort' command didn't work as expected")