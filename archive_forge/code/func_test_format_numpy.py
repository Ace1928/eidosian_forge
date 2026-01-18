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
def test_format_numpy():
    if not np:
        skip('numpy not installed.')
    for test in _tests:
        lhs = represent(test[0], basis=A, format='numpy')
        rhs = to_numpy(test[1])
        if isinstance(lhs, numpy_ndarray):
            assert (lhs == rhs).all()
        else:
            assert lhs == rhs