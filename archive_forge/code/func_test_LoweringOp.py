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
def test_LoweringOp():
    assert Dagger(a) == ad
    assert Commutator(a, ad).doit() == Integer(1)
    assert Commutator(a, N).doit() == a
    assert qapply(a * k) == (sqrt(k.n) * SHOKet(k.n - Integer(1))).expand()
    assert qapply(a * kz) == Integer(0)
    assert qapply(a * kf) == (sqrt(kf.n) * SHOKet(kf.n - Integer(1))).expand()
    assert a.rewrite('xp').doit() == Integer(1) / sqrt(Integer(2) * hbar * m * omega) * (I * Px + m * omega * X)
    for i in range(ndim - 1):
        assert a_rep[i, i + 1] == sqrt(i + 1)