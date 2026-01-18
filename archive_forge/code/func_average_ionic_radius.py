from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@property
def average_ionic_radius(self) -> FloatWithUnit:
    """Average ionic radius for element (with units). The average is taken
        over all oxidation states of the element for which data is present.
        """
    if 'Ionic radii' in self._data:
        radii = self._data['Ionic radii']
        radius = sum(radii.values()) / len(radii)
    else:
        radius = 0.0
    return FloatWithUnit(radius, 'ang')