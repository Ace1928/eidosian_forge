from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_geometry_from_IUCr_symbol(self, IUCr_symbol):
    """
        Returns the coordination geometry of the given IUCr symbol.

        Args:
            IUCr_symbol: The IUCr symbol of the coordination geometry.
        """
    for gg in self.cg_list:
        if gg.IUCr_symbol == IUCr_symbol:
            return gg
    raise LookupError(f'No coordination geometry found with IUCr symbol {IUCr_symbol!r}')