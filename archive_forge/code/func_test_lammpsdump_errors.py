import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
def test_lammpsdump_errors(fmt, lammpsdump):
    with pytest.raises(ValueError, match='Cannot determine atom types.*'):
        _ = fmt.parse_atoms(lammpsdump(have_element=False, have_type=False))
    with pytest.raises(ValueError, match='No atomic positions found in LAMMPS output'):
        _ = fmt.parse_atoms(lammpsdump(position_cols='unk_x unk_y unk_z'))