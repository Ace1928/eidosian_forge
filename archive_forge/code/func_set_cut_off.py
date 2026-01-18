from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def set_cut_off(self, value):
    self._cut_off = value
    if self.gradient_direction == 'ascent':
        cv = 2 * self.cut_off
    else:
        cv = 0
    if self.closed_edges:
        shape_old = self.density_grid.shape
        self.cell_origin += -(1.0 / np.array(shape_old)) @ self.cell
        self.density_grid = np.pad(self.density_grid, pad_width=(1,), mode='constant', constant_values=cv)
        shape_new = self.density_grid.shape
        s = np.array(shape_new) / np.array(shape_old)
        self.cell = self.cell @ np.diag(s)
    self.spacing = tuple(1.0 / np.array(self.density_grid.shape))
    scaled_verts, faces, _, _ = self.compute_mesh(self.density_grid, self.cut_off, self.spacing, self.gradient_direction)
    self.verts = scaled_verts
    self.faces = faces