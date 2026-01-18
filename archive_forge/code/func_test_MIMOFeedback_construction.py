from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix
from sympy.physics.control import (TransferFunction, Series, Parallel,
from sympy.testing.pytest import raises
def test_MIMOFeedback_construction():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s ** 3 - 1, s)
    tf3 = TransferFunction(s, s + 1, s)
    tf4 = TransferFunction(s, s ** 2 + 1, s)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf3, tf4], [tf1, tf2]])
    f1 = MIMOFeedback(tfm_1, tfm_2)
    assert f1.args == (tfm_1, tfm_2, -1)
    assert f1.sys1 == tfm_1
    assert f1.sys2 == tfm_2
    assert f1.var == s
    assert f1.sign == -1
    assert --f1 == f1
    f2 = MIMOFeedback(tfm_2, tfm_1, 1)
    assert f2.args == (tfm_2, tfm_1, 1)
    assert f2.sys1 == tfm_2
    assert f2.sys2 == tfm_1
    assert f2.var == s
    assert f2.sign == 1
    f3 = MIMOFeedback(tfm_1, MIMOSeries(tfm_3, tfm_2))
    assert f3.args == (tfm_1, MIMOSeries(tfm_3, tfm_2), -1)
    assert f3.sys1 == tfm_1
    assert f3.sys2 == MIMOSeries(tfm_3, tfm_2)
    assert f3.var == s
    assert f3.sign == -1
    mat = Matrix([[1, 1 / s], [0, 1]])
    sys1 = controller = TransferFunctionMatrix.from_Matrix(mat, s)
    f4 = MIMOFeedback(sys1, controller)
    assert f4.args == (sys1, controller, -1)
    assert f4.sys1 == f4.sys2 == sys1