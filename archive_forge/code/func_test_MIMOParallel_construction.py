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
def test_MIMOParallel_construction():
    tfm1 = TransferFunctionMatrix([[TF1], [TF2], [TF3]])
    tfm2 = TransferFunctionMatrix([[-TF3], [TF2], [TF1]])
    tfm3 = TransferFunctionMatrix([[TF1]])
    tfm4 = TransferFunctionMatrix([[TF2], [TF1], [TF3]])
    tfm5 = TransferFunctionMatrix([[TF1, TF2], [TF2, TF1]])
    tfm6 = TransferFunctionMatrix([[TF2, TF1], [TF1, TF2]])
    tfm7 = TransferFunctionMatrix.from_Matrix(Matrix([[1 / p]]), p)
    p8 = MIMOParallel(tfm1, tfm2)
    assert p8.args == (tfm1, tfm2)
    assert p8.var == s
    assert p8.shape == (p8.num_outputs, p8.num_inputs) == (3, 1)
    p9 = MIMOParallel(MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.args == (MIMOSeries(tfm3, tfm1), tfm2)
    assert p9.var == s
    assert p9.shape == (p9.num_outputs, p9.num_inputs) == (3, 1)
    p10 = MIMOParallel(tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.args == (tfm1, MIMOSeries(tfm3, tfm4), tfm2)
    assert p10.var == s
    assert p10.shape == (p10.num_outputs, p10.num_inputs) == (3, 1)
    p11 = MIMOParallel(tfm2, tfm1, tfm4)
    assert p11.args == (tfm2, tfm1, tfm4)
    assert p11.shape == (p11.num_outputs, p11.num_inputs) == (3, 1)
    p12 = MIMOParallel(tfm6, tfm5)
    assert p12.args == (tfm6, tfm5)
    assert p12.shape == (p12.num_outputs, p12.num_inputs) == (2, 2)
    p13 = MIMOParallel(tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.args == (tfm2, tfm4, MIMOSeries(-tfm3, tfm4), -tfm4)
    assert p13.shape == (p13.num_outputs, p13.num_inputs) == (3, 1)
    raises(TypeError, lambda: MIMOParallel(()))
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm2, TF1))
    raises(TypeError, lambda: MIMOParallel(tfm1, tfm3, tfm4))
    raises(ValueError, lambda: MIMOParallel(tfm3, tfm7))
    raises(TypeError, lambda: MIMOParallel(2, tfm1, tfm4))
    raises(TypeError, lambda: MIMOParallel(s ** 2 + p * s, -tfm4, tfm2))