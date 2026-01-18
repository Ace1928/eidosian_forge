import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def compare_qm_cell_and_pbc(qm_calc, mm_calc, bulk_at, test_size=4, expected_pbc=np.array([True, True, True]), buffer_width=5 * 3.61):
    """
    test qm cell shape and choice of pbc:
    make a non-periodic pdc in a direction
    if qm_radius + buffer is larger than the original cell
    keep the periodic cell otherwise i. e. if cell[i, i] > qm_radius + buffer
    the scenario is controlled by the test_size used to create at0
    as well as buffer_width.
    If the size of the at0 is larger than the r_qm + buffer + vacuum
    the cell stays periodic and the size is the same is original
    otherwise cell is non-periodic and size is different.
    """
    alat = bulk_at.cell[0, 0]
    at0 = bulk_at * test_size
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    R_QM = alat / np.sqrt(2.0) + 0.001
    qm_mask = r < R_QM
    qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc, buffer_width=buffer_width)
    qmmm.initialize_qm_buffer_mask(at0)
    qm_cluster = qmmm.get_qm_cluster(at0)
    assert all(qm_cluster.pbc == expected_pbc)
    if not all(expected_pbc):
        assert not all(qm_cluster.cell.lengths()[~expected_pbc] == at0.cell.lengths()[~expected_pbc])
    if any(expected_pbc):
        np.testing.assert_allclose(qm_cluster.cell.lengths()[expected_pbc], at0.cell.lengths()[expected_pbc])