import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def normalize_special_points(special_points):
    dct = {}
    for name, value in special_points.items():
        if not isinstance(name, str):
            raise TypeError('Expected name to be a string')
        if not np.shape(value) == (3,):
            raise ValueError('Expected 3 kpoint coordinates')
        dct[name] = np.asarray(value, float)
    return dct