import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
@pytest.mark.slow
def test_forceqmmm(qm_calc, mm_calc, bulk_at):
    N_cell = 2
    R_QMs = np.array([3, 7])
    sigma = (bulk_at * 2).get_distance(0, 1) * 2.0 ** (-1.0 / 6)
    at0 = bulk_at * N_cell
    r = at0.get_distances(0, np.arange(1, len(at0)), mic=True)
    print(len(r))
    del at0[0]
    print('N_cell', N_cell, 'N_MM', len(at0), 'Size', N_cell * bulk_at.cell[0, 0])
    ref_at = at0.copy()
    ref_at.calc = qm_calc
    opt = FIRE(ref_at)
    opt.run(fmax=0.001)
    u_ref = ref_at.positions - at0.positions
    us = []
    for R_QM in R_QMs:
        at = at0.copy()
        qm_mask = r < R_QM
        qm_buffer_mask_ref = r < 2 * qm_calc.rc + R_QM
        print(f'R_QM             {R_QM}   N_QM        {qm_mask.sum()}')
        print(f'R_QM + buffer: {2 * qm_calc.rc + R_QM:.2f} N_QM_buffer {qm_buffer_mask_ref.sum()}')
        print(f'                     N_total:    {len(at)}')
        qmmm = ForceQMMM(at, qm_mask, qm_calc, mm_calc, buffer_width=2 * qm_calc.rc)
        qmmm.initialize_qm_buffer_mask(at)
        at.calc = qmmm
        opt = FIRE(at)
        opt.run(fmax=0.001)
        us.append(at.positions - at0.positions)

    def strain_error(at0, u_ref, u, cutoff, mask):
        I, J = neighbor_list('ij', at0, cutoff)
        I, J = np.array([(i, j) for i, j in zip(I, J) if mask[i]]).T
        v = u_ref - u
        dv = np.linalg.norm(v[I, :] - v[J, :], axis=1)
        return np.linalg.norm(dv)
    du_global = [strain_error(at0, u_ref, u, 1.5 * sigma, np.ones(len(r))) for u in us]
    du_local = [strain_error(at0, u_ref, u, 1.5 * sigma, r < 3.0) for u in us]
    print('du_local', du_local)
    print('du_global', du_global)
    assert np.all(np.diff(du_local) < 0)
    assert np.all(np.diff(du_global) < 0)
    assert du_local[-1] < 1e-10
    assert du_global[-1] < 1e-10