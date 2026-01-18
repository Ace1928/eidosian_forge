from sympy.core.numbers import (Float, I, Integer)
from sympy.matrices.dense import Matrix
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.represent import (represent, rep_innerproduct,
from sympy.physics.quantum.state import Bra, Ket
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.matrixutils import (numpy_ndarray,
from sympy.physics.quantum.cartesian import XKet, XOp, XBra
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state
def test_format_scipy_sparse():
    if not np:
        skip('numpy not installed.')
    if not scipy:
        skip('scipy not installed.')
    for test in _tests:
        lhs = represent(test[0], basis=A, format='scipy.sparse')
        rhs = to_scipy_sparse(test[1])
        if isinstance(lhs, scipy_sparse_matrix):
            assert np.linalg.norm((lhs - rhs).todense()) == 0.0
        else:
            assert lhs == rhs