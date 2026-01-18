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
def test_bad_hessian2d(self, n2_data):
    bad_hessians = (None, 'fish', 1, np.array([1, 2, 3]), n2_data['hessian'], np.array([[[1, 0, 0]], [[0, 0, 1]]]))
    for bad_hessian in bad_hessians:
        with pytest.raises(ValueError):
            VibrationsData.from_2d(n2_data['atoms'], bad_hessian)