from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
def set_volumetric_data(self, grid, structure):
    """
        Will create the VolumetricData Objects.

        Args:
            grid: grid on which wavefunction was calculated, e.g. [1,2,2]
            structure: Structure object
        """
    Nx = grid[0] - 1
    Ny = grid[1] - 1
    Nz = grid[2] - 1
    a = structure.lattice.matrix[0]
    b = structure.lattice.matrix[1]
    c = structure.lattice.matrix[2]
    new_x = []
    new_y = []
    new_z = []
    new_real = []
    new_imaginary = []
    new_density = []
    runner = 0
    for x in range(Nx + 1):
        for y in range(Ny + 1):
            for z in range(Nz + 1):
                x_here = x / float(Nx) * a[0] + y / float(Ny) * b[0] + z / float(Nz) * c[0]
                y_here = x / float(Nx) * a[1] + y / float(Ny) * b[1] + z / float(Nz) * c[1]
                z_here = x / float(Nx) * a[2] + y / float(Ny) * b[2] + z / float(Nz) * c[2]
                if x != Nx and y != Ny and (z != Nz):
                    if not np.isclose(self.points[runner][0], x_here, 0.001) and (not np.isclose(self.points[runner][1], y_here, 0.001)) and (not np.isclose(self.points[runner][2], z_here, 0.001)):
                        raise ValueError('The provided wavefunction from Lobster does not contain all relevant points. Please use a line similar to: printLCAORealSpaceWavefunction kpoint 1 coordinates 0.0 0.0 0.0 coordinates 1.0 1.0 1.0 box bandlist 1 ')
                    new_x += [x_here]
                    new_y += [y_here]
                    new_z += [z_here]
                    new_real += [self.real[runner]]
                    new_imaginary += [self.imaginary[runner]]
                    new_density += [self.real[runner] ** 2 + self.imaginary[runner] ** 2]
                runner += 1
    self.final_real = np.reshape(new_real, [Nx, Ny, Nz])
    self.final_imaginary = np.reshape(new_imaginary, [Nx, Ny, Nz])
    self.final_density = np.reshape(new_density, [Nx, Ny, Nz])
    self.volumetricdata_real = VolumetricData(structure, {'total': self.final_real})
    self.volumetricdata_imaginary = VolumetricData(structure, {'total': self.final_imaginary})
    self.volumetricdata_density = VolumetricData(structure, {'total': self.final_density})