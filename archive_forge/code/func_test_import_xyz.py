import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def test_import_xyz(at0, qm_calc, mm_calc, testdir):
    """
    test the import_extxyz function and checks the mapping
    """
    filename = 'qmmm_export_test.xyz'
    qmmm = at0.calc
    qmmm.export_extxyz(filename=filename, atoms=at0)
    imported_qmmm = ForceQMMM.import_extxyz(filename, qm_calc, mm_calc)
    assert all(imported_qmmm.qm_selection_mask == qmmm.qm_selection_mask)
    assert all(imported_qmmm.qm_buffer_mask == qmmm.qm_buffer_mask)