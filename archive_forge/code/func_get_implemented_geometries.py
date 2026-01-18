from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def get_implemented_geometries(self, coordination=None, returned='cg', include_deactivated=False):
    """
        Returns a list of the implemented coordination geometries with the given coordination number.

        Args:
            coordination: The coordination number of which the list of implemented coordination geometries
                are returned.
            returned: Type of objects in the list.
            include_deactivated: Whether to include CoordinationGeometry that are deactivated.
        """
    geom = []
    if coordination is None:
        for gg in self.cg_list:
            if gg.points is not None and (not gg.deactivate or include_deactivated):
                if returned == 'cg':
                    geom.append(gg)
                elif returned == 'mp_symbol':
                    geom.append(gg.mp_symbol)
    else:
        for gg in self.cg_list:
            if gg.get_coordination_number() == coordination and gg.points is not None and (not gg.deactivate or include_deactivated):
                if returned == 'cg':
                    geom.append(gg)
                elif returned == 'mp_symbol':
                    geom.append(gg.mp_symbol)
    return geom