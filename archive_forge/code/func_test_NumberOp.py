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
def test_NumberOp():
    assert Commutator(N, ad).doit() == ad
    assert Commutator(N, a).doit() == Integer(-1) * a
    assert Commutator(N, H).doit() == Integer(0)
    assert qapply(N * k) == (k.n * k).expand()
    assert N.rewrite('a').doit() == ad * a
    assert N.rewrite('xp').doit() == Integer(1) / (Integer(2) * m * hbar * omega) * (Px ** 2 + (m * omega * X) ** 2) - Integer(1) / Integer(2)
    assert N.rewrite('H').doit() == H / (hbar * omega) - Integer(1) / Integer(2)
    for i in range(ndim):
        assert N_rep[i, i] == i
    assert N_rep == ad_rep_sympy * a_rep