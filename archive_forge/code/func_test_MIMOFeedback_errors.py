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
def test_MIMOFeedback_errors():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s ** 3 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s, s ** 2 + 1, s)
    tf5 = TransferFunction(1, 1, s)
    tf6 = TransferFunction(-1, s - 1, s)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf3], [tf4, tf1]])
    tfm_3 = TransferFunctionMatrix.from_Matrix(eye(2), var=s)
    tfm_4 = TransferFunctionMatrix([[tf1, tf5], [tf5, tf5]])
    tfm_5 = TransferFunctionMatrix([[-tf3, tf3], [tf3, tf6]])
    tfm_6 = TransferFunctionMatrix([[-tf3]])
    tfm_7 = TransferFunctionMatrix([[tf3, tf4]])
    raises(TypeError, lambda: MIMOFeedback(tf1, tf2))
    raises(TypeError, lambda: MIMOFeedback(MIMOParallel(tfm_1, tfm_2), tfm_3))
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_6, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_7, tfm_7))
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_2, -2))
    raises(ValueError, lambda: MIMOFeedback(tfm_5, tfm_4, 1))
    raises(ValueError, lambda: MIMOFeedback(tfm_4, -tfm_5))
    raises(ValueError, lambda: MIMOFeedback(tfm_3, tfm_3, 1))
    tfm_8 = TransferFunctionMatrix.from_Matrix(eye(2), var=p)
    raises(ValueError, lambda: MIMOFeedback(tfm_1, tfm_8, 1))