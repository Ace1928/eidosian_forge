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
def average_anionic_radius(self) -> float:
    """Average anionic radius for element (with units). The average is
        taken over all negative oxidation states of the element for which
        data is present.
        """
    if 'Ionic radii' in self._data:
        radii = [v for k, v in self._data['Ionic radii'].items() if int(k) < 0]
        if radii:
            return FloatWithUnit(sum(radii) / len(radii), 'ang')
    return FloatWithUnit(0.0, 'ang')