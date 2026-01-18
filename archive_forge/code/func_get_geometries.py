from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_geometries(self, coordination=None, returned='cg'):
    """
        Returns a list of coordination geometries with the given coordination number.

        Args:
            coordination: The coordination number of which the list of coordination geometries are returned.
            returned: Type of objects in the list.
        """
    geom = []
    if coordination is None:
        for gg in self.cg_list:
            if returned == 'cg':
                geom.append(gg)
            elif returned == 'mp_symbol':
                geom.append(gg.mp_symbol)
    else:
        for gg in self.cg_list:
            if gg.get_coordination_number() == coordination:
                if returned == 'cg':
                    geom.append(gg)
                elif returned == 'mp_symbol':
                    geom.append(gg.mp_symbol)
    return geom