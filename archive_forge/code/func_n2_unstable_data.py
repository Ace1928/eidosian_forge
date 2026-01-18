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
def n2_unstable_data(self):
    return {'atoms': Atoms('N2', positions=[[0.0, 0.0, 0.45], [0.0, 0.0, -0.45]]), 'hessian': np.array([-5.150829928323684, 0.0, -0.6867385017096544, 5.150829928323684, 0.0, 0.6867385017096544, 0.0, -5.158454318599951, 0.0, 0.0, 5.158454318599951, 0.0, -0.6867385017096544, 0.0, 56.65107699250456, 0.6867385017096544, 0.0, -56.65107699250456, 5.150829928323684, 0.0, 0.6867385017096544, -5.150829928323684, 0.0, -0.6867385017096544, 0.0, 5.158454318599951, 0.0, 0.0, -5.158454318599951, 0.0, 0.6867385017096544, 0.0, -56.65107699250456, -0.6867385017096544, 0.0, 56.65107699250456]).reshape((2, 3, 2, 3))}