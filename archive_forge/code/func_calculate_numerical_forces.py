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
def calculate_numerical_forces(self, atoms, d=0.001):
    """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""
    from ase.calculators.test import numeric_force
    return np.array([[numeric_force(atoms, a, i, d) for i in range(3)] for a in range(len(atoms))])