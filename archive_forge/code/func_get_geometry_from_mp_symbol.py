from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_geometry_from_mp_symbol(self, mp_symbol):
    """
        Returns the coordination geometry of the given mp_symbol.

        Args:
            mp_symbol: The mp_symbol of the coordination geometry.
        """
    for gg in self.cg_list:
        if gg.mp_symbol == mp_symbol:
            return gg
    raise LookupError(f'No coordination geometry found with mp_symbol {mp_symbol!r}')