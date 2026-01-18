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
def test_jmol_roundtrip(self, testdir, n2_data):
    ir_intensities = np.random.RandomState(42).rand(6)
    vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
    vib_data.write_jmol(self.jmol_file, ir_intensities=ir_intensities)
    images = ase.io.read(self.jmol_file, index=':')
    for i, image in enumerate(images):
        assert_array_almost_equal(image.positions, vib_data.get_atoms().positions)
        assert image.info['IR_intensity'] == pytest.approx(ir_intensities[i])
        assert_array_almost_equal(image.arrays['mode'], vib_data.get_modes()[i])