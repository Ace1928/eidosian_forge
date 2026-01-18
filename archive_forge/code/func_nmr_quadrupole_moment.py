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
def nmr_quadrupole_moment(self) -> dict[str, FloatWithUnit]:
    """Get a dictionary the nuclear electric quadrupole moment in units of
        e*millibarns for various isotopes.
        """
    return {k: FloatWithUnit(v, 'mbarn') for k, v in self.data.get('NMR Quadrupole Moment', {}).items()}