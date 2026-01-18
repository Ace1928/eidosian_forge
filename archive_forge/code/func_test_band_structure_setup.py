import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def test_band_structure_setup(testing_calculator):
    c = testing_calculator
    from ase.dft.kpoints import BandPath
    atoms = ase.build.bulk('Ag')
    bp = BandPath(cell=atoms.cell, path='GX', special_points={'G': [0, 0, 0], 'X': [0.5, 0, 0.5]})
    bp = bp.interpolate(npoints=10)
    c.set_bandpath(bp)
    kpt_list = c.cell.bs_kpoint_list.value.split('\n')
    assert len(kpt_list) == 10
    assert list(map(float, kpt_list[0].split())) == [0.0, 0.0, 0.0]
    assert list(map(float, kpt_list[-1].split())) == [0.5, 0.0, 0.5]