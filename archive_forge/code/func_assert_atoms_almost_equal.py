import os
import numpy as np
import numpy.testing
import unittest
import ase
import ase.build
import ase.io
from ase.io.vasp import write_vasp_xdatcar
from ase.calculators.calculator import compare_atoms
def assert_atoms_almost_equal(self, atoms, other, tol=1e-15):
    """Compare two Atoms objects, raising AssertionError if different"""
    system_changes = compare_atoms(atoms, other, tol=tol)
    if len(system_changes) > 0:
        raise AssertionError('Atoms objects differ by {}'.format(', '.join(system_changes)))