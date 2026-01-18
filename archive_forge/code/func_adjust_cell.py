import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def adjust_cell(self, atoms, cell):
    if not self.do_adjust_cell:
        return
    cur_cell = atoms.get_cell()
    cur_cell_inv = atoms.cell.reciprocal().T
    delta_deform_grad = np.dot(cur_cell_inv, cell).T - np.eye(3)
    max_delta_deform_grad = np.max(np.abs(delta_deform_grad))
    if max_delta_deform_grad > 0.25:
        raise RuntimeError('FixSymmetry adjust_cell does not work properly with large deformation gradient step {} > 0.25'.format(max_delta_deform_grad))
    elif max_delta_deform_grad > 0.15:
        warnings.warn('FixSymmetry adjust_cell may be ill behaved with large deformation gradient step {}'.format(max_delta_deform_grad))
    symmetrized_delta_deform_grad = symmetrize_rank2(cur_cell, cur_cell_inv, delta_deform_grad, self.rotations)
    cell[:] = np.dot(cur_cell, (symmetrized_delta_deform_grad + np.eye(3)).T)