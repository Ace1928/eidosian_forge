import pytest
import numpy as np
from ase.build import bulk
def test_abc_and_scaled_position(atoms):
    scaled = get_spos(atoms)
    for i, atom in enumerate(atoms):
        assert np.allclose(scaled[i], atom.scaled_position)
        assert np.allclose(scaled[i], [atom.a, atom.b, atom.c])