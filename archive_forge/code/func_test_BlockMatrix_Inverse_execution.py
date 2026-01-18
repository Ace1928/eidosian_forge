import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
@SKIP
def test_BlockMatrix_Inverse_execution():
    k, n = (2, 4)
    dtype = 'float32'
    A = sy.MatrixSymbol('A', n, k)
    B = sy.MatrixSymbol('B', n, n)
    inputs = (A, B)
    output = B.I * A
    cutsizes = {A: [(n // 2, n // 2), (k // 2, k // 2)], B: [(n // 2, n // 2), (n // 2, n // 2)]}
    cutinputs = [sy.blockcut(i, *cutsizes[i]) for i in inputs]
    cutoutput = output.subs(dict(zip(inputs, cutinputs)))
    dtypes = dict(zip(inputs, [dtype] * len(inputs)))
    f = theano_function_(inputs, [output], dtypes=dtypes, cache={})
    fblocked = theano_function_(inputs, [sy.block_collapse(cutoutput)], dtypes=dtypes, cache={})
    ninputs = [np.random.rand(*x.shape).astype(dtype) for x in inputs]
    ninputs = [np.arange(n * k).reshape(A.shape).astype(dtype), np.eye(n).astype(dtype)]
    ninputs[1] += np.ones(B.shape) * 1e-05
    assert np.allclose(f(*ninputs), fblocked(*ninputs), rtol=1e-05)