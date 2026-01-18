import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
def test_harmonic_vibrations(self, testdir):
    """Check the numerics with a trivial case: one atom in harmonic well"""
    rng = np.random.RandomState(42)
    k = rng.rand()
    ref_atoms = Atoms('H', positions=np.zeros([1, 3]))
    atoms = ref_atoms.copy()
    mass = atoms.get_masses()[0]
    atoms.calc = ForceConstantCalculator(D=np.eye(3) * k, ref=ref_atoms, f0=np.zeros((1, 3)))
    vib = Vibrations(atoms, name='harmonic')
    vib.run()
    vib.read()
    expected_energy = units._hbar * np.sqrt(k * units._e * units.m ** 2 / mass / units._amu) / units._e
    assert np.allclose(vib.get_energies(), expected_energy)