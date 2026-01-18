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
def ionization_energy(self) -> float | None:
    """First ionization energy of element."""
    if not self.ionization_energies:
        warnings.warn(f'No data available for ionization_energy for {self.symbol}')
        return None
    return self.ionization_energies[0]