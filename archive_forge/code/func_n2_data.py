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
@pytest.fixture
def n2_data(self):
    return {'atoms': Atoms('N2', positions=[[0.0, 0.0, 0.05095057], [0.0, 0.0, 1.04904943]]), 'hessian': np.array([[[[0.00467554672, 0.0, 0.0], [-0.00467554672, 0.0, 0.0]], [[0.0, 0.00467554672, 0.0], [0.0, -0.00467554672, 0.0]], [[0.0, 0.0, 39.0392599], [0.0, 0.0, -39.0392599]]], [[[-0.00467554672, 0.0, 0.0], [0.00467554672, 0.0, 0.0]], [[0.0, -0.00467554672, 0.0], [0.0, 0.00467554672, 0.0]], [[0.0, 0.0, -39.0392599], [0.0, 0.0, 39.0392599]]]]), 'ref_frequencies': [0.0 + 0j, 6.0677553e-08 + 0j, 3.62010442e-06 + 0j, 13.4737571 + 0j, 13.4737571 + 0j, 1231.18496 + 0j], 'ref_zpe': 0.07799427233401508, 'ref_forces': np.array([[0.0, 0.0, -0.226722], [0.0, 0.0, 0.226722]])}