from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
def test_interpolate_images_scaled_coord(images, initial):
    interpolate(images, use_scaled_coord=True)
    assert_interpolated([image.get_scaled_positions() for image in images])
    assert np.allclose(images[1].cell, initial.cell)