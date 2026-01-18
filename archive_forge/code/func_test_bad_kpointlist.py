import numpy as np
import pytest
from ase.lattice import MCLC
def test_bad_kpointlist(cell):
    with pytest.raises(ValueError):
        cell.bandpath([np.zeros(2)])