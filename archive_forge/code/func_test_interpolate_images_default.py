from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
def test_interpolate_images_default(images, initial, average_pos):
    interpolate(images)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert np.allclose(images[1].cell, initial.cell)