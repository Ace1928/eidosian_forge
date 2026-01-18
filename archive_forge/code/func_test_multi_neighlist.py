import numpy as np
from ase import Atoms
from pytest import mark
@mark.calculator_lite
def test_multi_neighlist(KIM, testdir):
    """
    To test that the correct energy/forces/stress can be computed using a
    model that implements multiple cutoffs.  This is done by construct a 10
    Angstrom x 10 Angstrom x 10 Angstrom non-periodic cell filled with 15
    randomly positioned atoms and requesting tha tthe model compute the
    energy, forces, and virial stress.  The energy returned by the model is
    compared to a known precomputed value, while the forces and stress
    returned are compared to numerical estimates via finite difference.
    """
    positions = np.random.RandomState(34).rand(15, 3) * 10
    atoms = Atoms('Ar' * 15, positions=positions, pbc=False, cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    calc = KIM('ex_model_Ar_P_Morse_MultiCutoff')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    energy_ref = 34.69963483186903
    forces_numer = calc.calculate_numerical_forces(atoms, d=0.0001)
    stress_numer = calc.calculate_numerical_stress(atoms, d=0.0001, voigt=True)
    tol = 1e-06
    assert np.isclose(energy, energy_ref, tol)
    assert np.allclose(forces, forces_numer, tol)
    assert np.allclose(stress, stress_numer, tol)