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
def get_crystal_field_spin(self, coordination: Literal['oct', 'tet']='oct', spin_config: Literal['low', 'high']='high') -> float:
    """Calculate the crystal field spin based on coordination and spin
        configuration. Only works for transition metal species.

        Args:
            coordination ("oct" | "tet"): Tetrahedron or octahedron crystal site coordination
            spin_config ("low" | "high"): Whether the species is in a high or low spin state

        Returns:
            Crystal field spin in Bohr magneton.

        Raises:
            AttributeError if species is not a valid transition metal or has
                an invalid oxidation state.
            ValueError if invalid coordination or spin_config.
        """
    if coordination not in ('oct', 'tet') or spin_config not in ('high', 'low'):
        raise ValueError('Invalid coordination or spin config')
    elec = self.full_electronic_structure
    if len(elec) < 4 or elec[-1][1] != 's' or elec[-2][1] != 'd':
        raise AttributeError(f'Invalid element {self.symbol} for crystal field calculation')
    n_electrons = elec[-1][2] + elec[-2][2] - self.oxi_state
    if n_electrons < 0 or n_electrons > 10:
        raise AttributeError(f'Invalid oxidation state {self.oxi_state} for element {self.symbol}')
    if spin_config == 'high':
        if n_electrons <= 5:
            return n_electrons
        return 10 - n_electrons
    if spin_config == 'low':
        if coordination == 'oct':
            if n_electrons <= 3:
                return n_electrons
            if n_electrons <= 6:
                return 6 - n_electrons
            if n_electrons <= 8:
                return n_electrons - 6
            return 10 - n_electrons
        if coordination == 'tet':
            if n_electrons <= 2:
                return n_electrons
            if n_electrons <= 4:
                return 4 - n_electrons
            if n_electrons <= 7:
                return n_electrons - 4
            return 10 - n_electrons
        return None
    return None