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
def kpts2ndarray(kpts, atoms=None):
    """Convert kpts keyword to 2-d ndarray of scaled k-points."""
    return kpts2kpts(kpts, atoms=atoms).kpts