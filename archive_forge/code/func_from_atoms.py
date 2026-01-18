from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
@classmethod
def from_atoms(cls, atoms, **kwargs):
    return cls.from_plotting_variables(PlottingVariables(atoms, scale=1.0), **kwargs)