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
def test_MIMOSeries_functions():
    tfm1 = TransferFunctionMatrix([[TF1, TF2, TF3], [-TF3, -TF2, TF1]])
    tfm2 = TransferFunctionMatrix([[-TF1], [-TF2], [-TF3]])
    tfm3 = TransferFunctionMatrix([[-TF1]])
    tfm4 = TransferFunctionMatrix([[-TF2, -TF3], [-TF1, TF2]])
    tfm5 = TransferFunctionMatrix([[TF2, -TF2], [-TF3, -TF2]])
    tfm6 = TransferFunctionMatrix([[-TF3], [TF1]])
    tfm7 = TransferFunctionMatrix([[TF1], [-TF2]])
    assert tfm1 * tfm2 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm6)
    assert tfm1 * tfm2 + tfm7 + tfm6 == MIMOParallel(MIMOSeries(tfm2, tfm1), tfm7, tfm6)
    assert tfm1 * tfm2 - tfm6 - tfm7 == MIMOParallel(MIMOSeries(tfm2, tfm1), -tfm6, -tfm7)
    assert tfm4 * tfm5 + (tfm4 - tfm5) == MIMOParallel(MIMOSeries(tfm5, tfm4), tfm4, -tfm5)
    assert tfm4 * -tfm6 + -tfm4 * tfm6 == MIMOParallel(MIMOSeries(-tfm6, tfm4), MIMOSeries(tfm6, -tfm4))
    raises(ValueError, lambda: tfm1 * tfm2 + TF1)
    raises(TypeError, lambda: tfm1 * tfm2 + a0)
    raises(TypeError, lambda: tfm4 * tfm6 - (s - 1))
    raises(TypeError, lambda: tfm4 * -tfm6 - 8)
    raises(TypeError, lambda: -1 + p ** 5 + tfm1 * tfm2)
    raises(TypeError, lambda: -tfm1 * tfm2 + tfm4)
    raises(TypeError, lambda: tfm1 * tfm2 - tfm4 + tfm5)
    raises(TypeError, lambda: tfm1 * tfm2 - tfm4 * tfm5)
    assert tfm1 * tfm2 * -tfm3 == MIMOSeries(-tfm3, tfm2, tfm1)
    assert tfm1 * -tfm2 * tfm3 == MIMOSeries(tfm3, -tfm2, tfm1)
    raises(ValueError, lambda: tfm4 * tfm5 * TF1)
    raises(TypeError, lambda: tfm4 * tfm5 * a1)
    raises(TypeError, lambda: tfm4 * -tfm5 * (s - 2))
    raises(TypeError, lambda: tfm5 * tfm4 * 9)
    raises(TypeError, lambda: (-p ** 3 + 1) * tfm5 * tfm4)
    assert MIMOSeries(tfm2, tfm1, evaluate=True) == MIMOSeries(tfm2, tfm1).doit() == TransferFunctionMatrix(((TransferFunction(-k ** 2 * (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (-a2 * p + s) * (a2 * p - s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 - (a2 * s + p) ** 2, (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2, s),), (TransferFunction(k ** 2 * (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (-a2 * p + s) * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + (a2 * p - s) * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2, s),)))
    mat_1 = Matrix([[1 / (1 + s), (1 + s) / (1 + s ** 2 + 2 * s) ** 3]])
    mat_2 = Matrix([[1 + s], [(1 + s ** 2 + 2 * s) ** 3 / (1 + s)]])
    tm_1, tm_2 = (TransferFunctionMatrix.from_Matrix(mat_1, s), TransferFunctionMatrix.from_Matrix(mat_2, s))
    assert MIMOSeries(tm_2, tm_1).doit() == TransferFunctionMatrix(((TransferFunction(2 * (s + 1) ** 2 * (s ** 2 + 2 * s + 1) ** 3, (s + 1) ** 2 * (s ** 2 + 2 * s + 1) ** 3, s),),))
    assert MIMOSeries(tm_2, tm_1).doit().simplify() == TransferFunctionMatrix(((TransferFunction(2, 1, s),),))
    assert MIMOSeries(-tfm3, -tfm2, tfm1, evaluate=True) == MIMOSeries(-tfm3, -tfm2, tfm1).doit() == TransferFunctionMatrix(((TransferFunction(k ** 2 * (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (a2 * p - s) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (a2 * s + p) ** 2, (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 3, s),), (TransferFunction(-k ** 2 * (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 2 + (-a2 * p + s) * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2) + (a2 * p - s) * (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), (a2 * s + p) ** 2 * (s ** 2 + 2 * s * wn * zeta + wn ** 2) ** 3, s),)))
    assert MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5, evaluate=True) == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).doit() == TransferFunctionMatrix(((TransferFunction(-k * (-a2 * s - p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s), TransferFunction(k * (-a2 * p - k * (a2 * s + p) + s), a2 * s + p, s)), (TransferFunction(-k * (-a2 * s - p + (-a2 * p + s) * (s ** 2 + 2 * s * wn * zeta + wn ** 2)), (a2 * s + p) * (s ** 2 + 2 * s * wn * zeta + wn ** 2), s), TransferFunction((-a2 * p + s) * (-a2 * p - k * (a2 * s + p) + s), (a2 * s + p) ** 2, s)))) == MIMOSeries(MIMOParallel(tfm4, tfm5), tfm5).rewrite(TransferFunctionMatrix)