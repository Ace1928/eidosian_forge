import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def runtest_autowrap_twice(language, backend):
    f = autowrap((((a + b) / c) ** 5).expand(), language, backend)
    g = autowrap((((a + b) / c) ** 4).expand(), language, backend)
    assert f(1, -2, 1) == -1.0
    assert g(1, -2, 1) == 1.0