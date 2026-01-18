from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
@staticmethod
def compute_mesh(density_grid, cut_off, spacing, gradient_direction):
    """

        Import statement is in this method and not file header
        since few users will use isosurface rendering.

        Returns scaled_verts, faces, normals, values. See skimage docs.

        """
    from skimage import measure
    return measure.marching_cubes_lewiner(density_grid, level=cut_off, spacing=spacing, gradient_direction=gradient_direction, allow_degenerate=False)