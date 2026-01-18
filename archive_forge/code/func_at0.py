import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
@pytest.fixture
def at0(qm_calc, mm_calc, bulk_at):
    alat = bulk_at.cell[0, 0]
    at0 = bulk_at * 5
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    R_QM = alat / np.sqrt(2.0) + 0.001
    qm_mask = r < R_QM
    qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc, buffer_width=3.61)
    qmmm.initialize_qm_buffer_mask(at0)
    at0.calc = qmmm
    return at0