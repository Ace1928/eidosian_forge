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
def ionic_radii(self) -> dict[int, float]:
    """All ionic radii of the element as a dict of
        {oxidation state: ionic radii}. Radii are given in angstrom.
        """
    if 'Ionic radii' in self._data:
        return {int(k): FloatWithUnit(v, 'ang') for k, v in self._data['Ionic radii'].items()}
    return {}