import pytest
import numpy as np
from ase.lattice import (get_lattice_from_canonical_cell, all_variants,
@pytest.mark.parametrize('lat', variants)
def test_lattice(lat):
    cell = lat.tocell()

    def check(lat1):
        print('check', repr(lat), '-->', repr(lat1))
        err = np.abs(cell.cellpar() - lat1.cellpar()).max()
        assert err < 1e-05, err
    check(get_lattice_from_canonical_cell(cell))
    if lat.name == 'TRI':
        return
    stdcell, op = identify_lattice(cell, 0.0001)
    check(stdcell)
    rcell, op = cell.niggli_reduce()
    stdcell, op = identify_lattice(rcell, 0.0001)
    check(stdcell)