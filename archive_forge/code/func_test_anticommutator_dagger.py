from sympy.core.numbers import Integer
from sympy.core.symbol import symbols
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.anticommutator import AntiCommutator as AComm
from sympy.physics.quantum.operator import Operator
def test_anticommutator_dagger():
    assert Dagger(AComm(A, B)) == AComm(Dagger(A), Dagger(B))