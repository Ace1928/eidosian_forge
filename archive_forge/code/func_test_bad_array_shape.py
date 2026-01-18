import pytest
import numpy as np
from ase import Atoms
def test_bad_array_shape():
    with pytest.raises(ValueError, match='wrong length'):
        Atoms().set_masses([1, 2])
    with pytest.raises(ValueError, match='wrong length'):
        Atoms('H').set_masses([])
    with pytest.raises(ValueError, match='wrong shape'):
        Atoms('H').set_masses(np.ones((1, 3)))