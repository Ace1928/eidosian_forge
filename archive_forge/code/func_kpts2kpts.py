import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def kpts2kpts(kpts, atoms=None):
    from ase.dft.kpoints import monkhorst_pack
    if kpts is None:
        return KPoints()
    if hasattr(kpts, 'kpts'):
        return kpts
    if isinstance(kpts, dict):
        if 'kpts' in kpts:
            return KPoints(kpts['kpts'])
        if 'path' in kpts:
            cell = Cell.ascell(atoms.cell)
            return cell.bandpath(pbc=atoms.pbc, **kpts)
        size, offsets = kpts2sizeandoffsets(atoms=atoms, **kpts)
        return KPoints(monkhorst_pack(size) + offsets)
    if isinstance(kpts[0], int):
        return KPoints(monkhorst_pack(kpts))
    return KPoints(np.array(kpts))