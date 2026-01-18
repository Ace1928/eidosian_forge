import pytest
import numpy as np
from ase.build import molecule
@calc('abinit', chksymtnons=0)
@calc('cp2k')
@calc('espresso', tprnfor=True)
@calc('gpaw', mode='pw', symmetry='off', txt=None)
@calc('nwchem')
@calc('siesta')
def test_h2_bond(factory, atoms):
    d0 = atoms.get_distance(0, 1)
    atoms.calc = factory.calc()
    X = d0 + np.linspace(-0.08, 0.08, 5)
    E = []
    F = []
    for x in X:
        atoms.positions[1, 2] = x
        e = atoms.get_potential_energy(force_consistent=True)
        f = atoms.get_forces()
        E.append(e)
        F.append(f[1, 2])
    E = np.array(E)
    F = np.array(F)
    a, b, c = np.polyfit(X, E, 2)
    xmin = -b / (2.0 * a)
    fa, fb = np.polyfit(X, F, 1)
    k_from_energy = 2 * a
    k_from_forces = -fa
    assert xmin == pytest.approx(0.77, rel=0.05)
    assert k_from_energy == pytest.approx(k_from_forces, rel=0.05)
    assert k_from_energy == pytest.approx(k_refs.get(factory.name, k_ref_0), rel=0.05)