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
def test_fixed_atoms(self, n2_data):
    vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'][1:, :, 1:, :], indices=[1])
    assert vib_data.get_indices() == [1]
    assert vib_data.get_mask().tolist() == [False, True]