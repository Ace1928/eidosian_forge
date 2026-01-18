from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import Commutator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.represent import represent
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.sho1d import (RaisingOp, LoweringOp,
def test_Hamiltonian():
    assert Commutator(H, N).doit() == Integer(0)
    assert qapply(H * k) == (hbar * omega * (k.n + Integer(1) / Integer(2)) * k).expand()
    assert H.rewrite('a').doit() == hbar * omega * (ad * a + Integer(1) / Integer(2))
    assert H.rewrite('xp').doit() == Integer(1) / (Integer(2) * m) * (Px ** 2 + (m * omega * X) ** 2)
    assert H.rewrite('N').doit() == hbar * omega * (N + Integer(1) / Integer(2))
    for i in range(ndim):
        assert H_rep[i, i] == hbar * omega * (i + Integer(1) / Integer(2))