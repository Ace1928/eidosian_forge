from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def pop_deprecated(dct, name):
    import warnings
    if name in dct:
        del dct[name]
        warnings.warn(f'The "{name}" keyword of write_pov() is deprecated and has no effect; this will raise an error in the future.', FutureWarning)