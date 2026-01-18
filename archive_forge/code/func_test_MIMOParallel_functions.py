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
def test_MIMOParallel_functions():
    tf4 = TransferFunction(a0 * p + p ** a1 - s, p, p)
    tf5 = TransferFunction(a1 * s ** 2 + a2 * s - a0, s + a0, s)
    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF2], [tf5], [-TF1]])
    tfm3 = TransferFunctionMatrix([[tf5], [-tf5], [TF2]])
    tfm4 = TransferFunctionMatrix([[TF2, -tf5], [TF1, tf5]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF3, -tf5]])
    tfm6 = TransferFunctionMatrix([[-TF2]])
    tfm7 = TransferFunctionMatrix([[tf4], [-tf4], [tf4]])
    assert tfm1 + tfm2 + tfm3 == MIMOParallel(tfm1, tfm2, tfm3) == MIMOParallel(MIMOParallel(tfm1, tfm2), tfm3)
    assert tfm2 - tfm1 - tfm3 == MIMOParallel(tfm2, -tfm1, -tfm3)
    assert tfm2 - tfm3 + -tfm1 * tfm6 * -tfm6 == MIMOParallel(tfm2, -tfm3, MIMOSeries(-tfm6, tfm6, -tfm1))
    assert tfm1 + tfm1 - -tfm1 * tfm6 == MIMOParallel(tfm1, tfm1, -MIMOSeries(tfm6, -tfm1))
    assert tfm2 - tfm3 - tfm1 + tfm2 == MIMOParallel(tfm2, -tfm3, -tfm1, tfm2)
    assert tfm1 + tfm2 - tfm3 - tfm1 == MIMOParallel(tfm1, tfm2, -tfm3, -tfm1)
    raises(ValueError, lambda: tfm1 + tfm2 + TF2)
    raises(TypeError, lambda: tfm1 - tfm2 - a1)
    raises(TypeError, lambda: tfm2 - tfm3 - (s - 1))
    raises(TypeError, lambda: -tfm3 - tfm2 - 9)
    raises(TypeError, lambda: 1 - p ** 3 - tfm3 - tfm2)
    raises(ValueError, lambda: tfm3 - tfm2 - tfm7)
    raises(ValueError, lambda: tfm2 - tfm1 + tfm7)
    raises(TypeError, lambda: tfm1 + tfm2 + tfm4)
    raises(TypeError, lambda: tfm1 - tfm2 - tfm4)
    assert (tfm1 + tfm2) * tfm6 == MIMOSeries(tfm6, MIMOParallel(tfm1, tfm2))
    assert (tfm2 - tfm3) * tfm6 * -tfm6 == MIMOSeries(-tfm6, tfm6, MIMOParallel(tfm2, -tfm3))
    assert (tfm2 - tfm1 - tfm3) * (tfm6 + tfm6) == MIMOSeries(MIMOParallel(tfm6, tfm6), MIMOParallel(tfm2, -tfm1, -tfm3))
    raises(ValueError, lambda: (tfm4 + tfm5) * TF1)
    raises(TypeError, lambda: (tfm2 - tfm3) * a2)
    raises(TypeError, lambda: (tfm3 + tfm2) * (s - 6))
    raises(TypeError, lambda: (tfm1 + tfm2 + tfm3) * 0)
    raises(TypeError, lambda: (1 - p ** 3) * (tfm1 + tfm3))
    raises(ValueError, lambda: (tfm3 - tfm2) * tfm4 * tfm5)
    raises(ValueError, lambda: (tfm1 - tfm2) * tfm5)
    assert MIMOParallel(tfm1, tfm2, evaluate=True) == MIMOParallel(tfm1, tfm2).doit() == MIMOParallel(tfm1, tfm2).rewrite(TransferFunctionMatrix) == TransferFunctionMatrix(((TransferFunction(-k * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + 1, s ** 2 + 2 * s * wn * zeta + wn ** 2, s),), (TransferFunction(-a0 + a1 * s ** 2 + a2 * s + k * (a0 + s), a0 + s, s),), (TransferFunction(-a2 * s - p + (a2 * p - s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s),)))