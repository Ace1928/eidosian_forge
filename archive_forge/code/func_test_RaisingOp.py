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
def test_RaisingOp():
    assert Dagger(ad) == a
    assert Commutator(ad, a).doit() == Integer(-1)
    assert Commutator(ad, N).doit() == Integer(-1) * ad
    assert qapply(ad * k) == (sqrt(k.n + 1) * SHOKet(k.n + 1)).expand()
    assert qapply(ad * kz) == (sqrt(kz.n + 1) * SHOKet(kz.n + 1)).expand()
    assert qapply(ad * kf) == (sqrt(kf.n + 1) * SHOKet(kf.n + 1)).expand()
    assert ad.rewrite('xp').doit() == Integer(1) / sqrt(Integer(2) * hbar * m * omega) * (Integer(-1) * I * Px + m * omega * X)
    assert ad.hilbert_space == ComplexSpace(S.Infinity)
    for i in range(ndim - 1):
        assert ad_rep_sympy[i + 1, i] == sqrt(i + 1)
    if not np:
        skip('numpy not installed.')
    ad_rep_numpy = represent(ad, basis=N, ndim=4, format='numpy')
    for i in range(ndim - 1):
        assert ad_rep_numpy[i + 1, i] == float(sqrt(i + 1))
    if not np:
        skip('numpy not installed.')
    if not scipy:
        skip('scipy not installed.')
    ad_rep_scipy = represent(ad, basis=N, ndim=4, format='scipy.sparse', spmatrix='lil')
    for i in range(ndim - 1):
        assert ad_rep_scipy[i + 1, i] == float(sqrt(i + 1))
    assert ad_rep_numpy.dtype == 'float64'
    assert ad_rep_scipy.dtype == 'float64'