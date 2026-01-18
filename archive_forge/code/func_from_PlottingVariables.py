from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
@classmethod
def from_PlottingVariables(cls, pvars, **kwargs):
    cell = pvars.cell
    cell_vertices = pvars.cell_vertices
    if 'colors' in kwargs.keys():
        colors = kwargs.pop('colors')
    else:
        colors = pvars.colors
    diameters = pvars.d
    image_height = pvars.h
    image_width = pvars.w
    positions = pvars.positions
    constraints = pvars.constraints
    return cls(cell=cell, cell_vertices=cell_vertices, colors=colors, constraints=constraints, diameters=diameters, image_height=image_height, image_width=image_width, positions=positions, **kwargs)