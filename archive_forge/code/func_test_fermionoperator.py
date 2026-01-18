from sympy.physics.quantum import Dagger, AntiCommutator, qapply
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.fermion import FermionFockKet, FermionFockBra
def test_fermionoperator():
    c = FermionOp('c')
    d = FermionOp('d')
    assert isinstance(c, FermionOp)
    assert isinstance(Dagger(c), FermionOp)
    assert c.is_annihilation
    assert not Dagger(c).is_annihilation
    assert FermionOp('c') == FermionOp('c', True)
    assert FermionOp('c') != FermionOp('d')
    assert FermionOp('c', True) != FermionOp('c', False)
    assert AntiCommutator(c, Dagger(c)).doit() == 1
    assert AntiCommutator(c, Dagger(d)).doit() == c * Dagger(d) + Dagger(d) * c