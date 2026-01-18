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
def test_MIMOFeedback_functions():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s - 1, s)
    tf3 = TransferFunction(1, 1, s)
    tf4 = TransferFunction(-1, s - 1, s)
    tfm_1 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf3]])
    tfm_3 = TransferFunctionMatrix([[-tf2, tf2], [tf2, tf4]])
    tfm_4 = TransferFunctionMatrix([[tf1, tf2], [-tf2, tf1]])
    F_1 = MIMOFeedback(tfm_2, tfm_3)
    F_2 = MIMOFeedback(tfm_2, MIMOSeries(tfm_4, -tfm_1), 1)
    assert F_1.sensitivity == Matrix([[S.Half, 0], [0, S.Half]])
    assert F_2.sensitivity == Matrix([[(-2 * s ** 4 + s ** 2) / (s ** 2 - s + 1), (2 * s ** 3 - s ** 2) / (s ** 2 - s + 1)], [-s ** 2, s]])
    assert F_1.doit() == TransferFunctionMatrix(((TransferFunction(1, 2 * s, s), TransferFunction(1, 2, s)), (TransferFunction(1, 2, s), TransferFunction(1, 2, s)))) == F_1.rewrite(TransferFunctionMatrix)
    assert F_2.doit(cancel=False, expand=True) == TransferFunctionMatrix(((TransferFunction(-s ** 5 + 2 * s ** 4 - 2 * s ** 3 + s ** 2, s ** 5 - 2 * s ** 4 + 3 * s ** 3 - 2 * s ** 2 + s, s), TransferFunction(-2 * s ** 4 + 2 * s ** 3, s ** 2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s ** 2 + s, 1, s))))
    assert F_2.doit(cancel=False) == TransferFunctionMatrix(((TransferFunction(s * (2 * s ** 3 - s ** 2) * (s ** 2 - s + 1) + (-2 * s ** 4 + s ** 2) * (s ** 2 - s + 1), s * (s ** 2 - s + 1) ** 2, s), TransferFunction(-2 * s ** 4 + 2 * s ** 3, s ** 2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s ** 2 + s, 1, s))))
    assert F_2.doit() == TransferFunctionMatrix(((TransferFunction(s * (-2 * s ** 2 + s * (2 * s - 1) + 1), s ** 2 - s + 1, s), TransferFunction(-2 * s ** 3 * (s - 1), s ** 2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(s * (1 - s), 1, s))))
    assert F_2.doit(expand=True) == TransferFunctionMatrix(((TransferFunction(-s ** 2 + s, s ** 2 - s + 1, s), TransferFunction(-2 * s ** 4 + 2 * s ** 3, s ** 2 - s + 1, s)), (TransferFunction(0, 1, s), TransferFunction(-s ** 2 + s, 1, s))))
    assert -F_1.doit() == (-F_1).doit()