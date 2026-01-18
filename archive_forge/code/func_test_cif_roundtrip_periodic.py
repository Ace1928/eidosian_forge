import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_roundtrip_periodic(atoms):
    atoms1 = roundtrip(atoms)
    assert str(atoms1.symbols) == 'CO'
    assert all(atoms1.pbc)
    assert atoms.cell.cellpar() == pytest.approx(atoms1.cell.cellpar(), abs=1e-05)
    assert atoms.get_scaled_positions() == pytest.approx(atoms1.get_scaled_positions(), abs=1e-05)