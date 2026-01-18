from sympy.simplify.epathtools import epath, EPath
from sympy.testing.pytest import raises
from sympy.core.numbers import E
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x, y, z, t
def test_EPath():
    assert EPath('/*/[0]')._path == '/*/[0]'
    assert EPath(EPath('/*/[0]'))._path == '/*/[0]'
    assert isinstance(epath('/*/[0]'), EPath) is True
    assert repr(EPath('/*/[0]')) == "EPath('/*/[0]')"
    raises(ValueError, lambda: EPath(''))
    raises(ValueError, lambda: EPath('/'))
    raises(ValueError, lambda: EPath('/|x'))
    raises(ValueError, lambda: EPath('/['))
    raises(ValueError, lambda: EPath('/[0]%'))
    raises(NotImplementedError, lambda: EPath('Symbol'))