from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_deprecated_find_executable():
    with warns_deprecated_sympy():
        find_executable('python')