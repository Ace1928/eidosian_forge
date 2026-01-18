from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_implicit_imports_regular_expression():
    candidates_ok = ['from sympy import something', '>>> from sympy import something', 'from sympy.somewhere import something', '>>> from sympy.somewhere import something', 'import sympy', '>>> import sympy', 'import sympy.something.something', '... import sympy', '... import sympy.something.something', '... from sympy import something', '... from sympy.somewhere import something', '>> from sympy import *', '# from sympy import *', 'some text # from sympy import *']
    candidates_fail = ['from sympy import *', '>>> from sympy import *', 'from sympy.somewhere import *', '>>> from sympy.somewhere import *', '... from sympy import *', '... from sympy.somewhere import *']
    for c in candidates_ok:
        assert implicit_test_re.search(_with_space(c)) is None, c
    for c in candidates_fail:
        assert implicit_test_re.search(_with_space(c)) is not None, c