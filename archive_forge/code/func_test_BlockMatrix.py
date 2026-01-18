import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def test_BlockMatrix():
    n = sy.Symbol('n', integer=True)
    A, B, C, D = [sy.MatrixSymbol(name, n, n) for name in 'ABCD']
    At, Bt, Ct, Dt = map(theano_code_, (A, B, C, D))
    Block = sy.BlockMatrix([[A, B], [C, D]])
    Blockt = theano_code_(Block)
    solutions = [tt.join(0, tt.join(1, At, Bt), tt.join(1, Ct, Dt)), tt.join(1, tt.join(0, At, Ct), tt.join(0, Bt, Dt))]
    assert any((theq(Blockt, solution) for solution in solutions))