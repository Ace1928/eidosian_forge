from sympy.core.numbers import Integer
from sympy.core.symbol import symbols
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.operator import Operator
def test_commutator_dagger():
    comm = Comm(A * B, C)
    assert Dagger(comm).expand(commutator=True) == -Comm(Dagger(B), Dagger(C)) * Dagger(A) - Dagger(B) * Comm(Dagger(A), Dagger(C))