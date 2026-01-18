from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_as_int():
    raises(ValueError, lambda: as_int(True))
    raises(ValueError, lambda: as_int(1.1))
    raises(ValueError, lambda: as_int([]))
    raises(ValueError, lambda: as_int(S.NaN))
    raises(ValueError, lambda: as_int(S.Infinity))
    raises(ValueError, lambda: as_int(S.NegativeInfinity))
    raises(ValueError, lambda: as_int(S.ComplexInfinity))
    raises(ValueError, lambda: as_int(1e+23))
    raises(ValueError, lambda: as_int(S('1.' + '0' * 20 + '1')))
    assert as_int(True, strict=False) == 1