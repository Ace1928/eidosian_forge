import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.mark.parametrize('x, weights, info, error', [(np.linspace(1, 10, 12), [np.linspace(4, 1, 12), np.sin(range(12))], [{'entry': '1'}, {'entry': '2'}], None), (np.linspace(1, 5, 7), [np.sqrt(range(7))], [{'entry': '1'}], None), (np.linspace(1, 5, 7), [np.ones((3, 3))], None, IndexError), (np.linspace(1, 5, 7), np.array([]).reshape(0, 7), None, IndexError), (np.linspace(1, 5, 7), np.ones((2, 6)), None, IndexError)])
def test_from_data(self, x, weights, info, error):
    if error is not None:
        with pytest.raises(error):
            dc = GridDOSCollection.from_data(x, weights, info=info)
    else:
        dc = GridDOSCollection.from_data(x, weights, info=info)
        for i, dos_data in enumerate(dc):
            assert dos_data.info == info[i]
            assert np.allclose(dos_data.get_energies(), x)
            assert np.allclose(dos_data.get_weights(), weights[i])