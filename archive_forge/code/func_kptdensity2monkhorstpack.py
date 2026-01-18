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
def kptdensity2monkhorstpack(atoms, kptdensity=3.5, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    atoms: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float
        Required k-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round up to even numbers.
    """
    recipcell = atoms.cell.reciprocal()
    kpts = []
    for i in range(3):
        if atoms.pbc[i]:
            k = 2 * pi * sqrt((recipcell[i] ** 2).sum()) * kptdensity
            if even:
                kpts.append(2 * int(np.ceil(k / 2)))
            else:
                kpts.append(int(np.ceil(k)))
        else:
            kpts.append(1)
    return np.array(kpts)