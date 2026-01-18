from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def require_pov(path):
    path = Path(path)
    if path.suffix != '.pov':
        raise ValueError(f'Expected .pov path, got {path}')
    return path