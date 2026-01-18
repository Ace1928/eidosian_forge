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
def max_oxidation_state(self) -> float:
    """Maximum oxidation state for element."""
    if 'Oxidation states' in self._data:
        return max(self._data['Oxidation states'])
    return 0