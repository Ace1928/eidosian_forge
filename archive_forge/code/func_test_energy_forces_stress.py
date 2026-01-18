import numpy as np
from pytest import mark
from ase.lattice.cubic import FaceCenteredCubic
@mark.calculator_lite
def test_energy_forces_stress(KIM, testdir):
    """
    To test that the calculator can produce correct energy and forces.  This
    is done by comparing the energy for an FCC argon lattice with an example
    model to the known value; the forces/stress returned by the model are
    compared to numerical estimates via finite difference.
    """
    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 1), symbol='Ar', pbc=(1, 0, 0), latticeconstant=3.0)
    atoms.positions[0, 0] += 0.01
    calc = KIM('ex_model_Ar_P_Morse_07C')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    energy_ref = 19.7196709065
    forces_numer = calc.calculate_numerical_forces(atoms, d=0.0001)
    stress_numer = calc.calculate_numerical_stress(atoms, d=0.0001, voigt=True)
    tol = 1e-06
    assert np.isclose(energy, energy_ref, tol)
    assert np.allclose(forces, forces_numer, tol)
    assert np.allclose(stress, stress_numer, tol)
    atoms.set_pbc(True)
    atoms.get_potential_energy()