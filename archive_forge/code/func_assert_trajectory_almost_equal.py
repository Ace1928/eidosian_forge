import os
import numpy as np
import numpy.testing
import unittest
import ase
import ase.build
import ase.io
from ase.io.vasp import write_vasp_xdatcar
from ase.calculators.calculator import compare_atoms
def assert_trajectory_almost_equal(self, traj1, traj2):
    self.assertEqual(len(traj1), len(traj2))
    for image, other in zip(traj1, traj2):
        self.assert_atoms_almost_equal(image, other)