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
def full_electronic_structure(self) -> list[tuple[int, str, int]]:
    """Full electronic structure as tuple.
        E.g., The electronic structure for Fe is represented as:
        [(1, "s", 2), (2, "s", 2), (2, "p", 6), (3, "s", 2), (3, "p", 6),
        (3, "d", 6), (4, "s", 2)].
        """
    e_str = self.electronic_structure

    def parse_orbital(orb_str):
        m = re.match('(\\d+)([spdfg]+)(\\d+)', orb_str)
        if m:
            return (int(m.group(1)), m.group(2), int(m.group(3)))
        return orb_str
    data = [parse_orbital(s) for s in e_str.split('.')]
    if data[0][0] == '[':
        sym = data[0].replace('[', '').replace(']', '')
        data = list(Element(sym).full_electronic_structure) + data[1:]
    return data