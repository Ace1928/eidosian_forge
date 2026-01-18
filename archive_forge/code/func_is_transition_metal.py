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
def is_transition_metal(self) -> bool:
    """True if element is a transition metal."""
    ns = (*range(21, 31), *range(39, 49), 57, *range(72, 81), 89, *range(104, 113))
    return self.Z in ns