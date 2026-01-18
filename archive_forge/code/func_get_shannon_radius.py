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
def get_shannon_radius(self, cn: str, spin: Literal['', 'Low Spin', 'High Spin']='', radius_type: Literal['ionic', 'crystal']='ionic') -> float:
    """Get the local environment specific ionic radius for species.

        Args:
            cn (str): Coordination using roman letters. Supported values are
                I-IX, as well as IIIPY, IVPY and IVSQ.
            spin (str): Some species have different radii for different
                spins. You can get specific values using "High Spin" or
                "Low Spin". Leave it as "" if not available. If only one spin
                data is available, it is returned and this spin parameter is
                ignored.
            radius_type (str): Either "crystal" or "ionic" (default).

        Returns:
            Shannon radius for specie in the specified environment.
        """
    radii = self._el.data['Shannon radii']
    radii = radii[str(int(self._oxi_state))][cn]
    if len(radii) == 1:
        key, data = next(iter(radii.items()))
        if key != spin:
            warnings.warn(f'Specified spin={spin!r} not consistent with database spin of {key}. Only one spin data available, and that value is returned.')
    else:
        data = radii[spin]
    return data[f'{radius_type}_radius']