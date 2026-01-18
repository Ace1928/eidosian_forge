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
def atomic_orbitals_eV(self) -> dict[str, float]:
    """
        Get the LDA energies in eV for neutral atoms, by orbital.

        This property contains the same info as `self.atomic_orbitals`,
        but uses eV for units, per matsci issue https://matsci.org/t/unit-of-atomic-orbitals-energy/54325
        In short, self.atomic_orbitals was meant to be in eV all along but is now kept
        as Hartree for backwards compatibility.
        """
    return {orb_idx: energy * Ha_to_eV for orb_idx, energy in self.atomic_orbitals.items()}