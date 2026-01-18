from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def get_extreme(self, target_prop, maximize=True, min_temp=None, max_temp=None, min_doping=None, max_doping=None, isotropy_tolerance=0.05, use_average=True):
    """This method takes in eigenvalues over a range of carriers,
        temperatures, and doping levels, and tells you what is the "best"
        value that can be achieved for the given target_property. Note that
        this method searches the doping dict only, not the full mu dict.

        Args:
            target_prop: target property, i.e. "seebeck", "power factor",
                         "conductivity", "kappa", or "zt"
            maximize: True to maximize, False to minimize (e.g. kappa)
            min_temp: minimum temperature allowed
            max_temp: maximum temperature allowed
            min_doping: minimum doping allowed (e.g., 1E18)
            max_doping: maximum doping allowed (e.g., 1E20)
            isotropy_tolerance: tolerance for isotropic (0.05 = 5%)
            use_average: True for avg of eigenval, False for max eigenval

        Returns:
            A dictionary with keys {"p", "n", "best"} with sub-keys:
            {"value", "temperature", "doping", "isotropic"}
        """

    def is_isotropic(x, isotropy_tolerance) -> bool:
        """Internal method to tell you if 3-vector "x" is isotropic.

            Args:
                x: the vector to determine isotropy for
                isotropy_tolerance: tolerance, e.g. 0.05 is 5%
            """
        if len(x) != 3:
            raise ValueError('Invalid input to is_isotropic!')
        st = sorted(x)
        return bool(all([st[0], st[1], st[2]]) and abs((st[1] - st[0]) / st[1]) <= isotropy_tolerance and (abs(st[2] - st[0]) / st[2] <= isotropy_tolerance) and (abs((st[2] - st[1]) / st[2]) <= isotropy_tolerance))
    if target_prop.lower() == 'seebeck':
        d = self.get_seebeck(output='eigs', doping_levels=True)
    elif target_prop.lower() == 'power factor':
        d = self.get_power_factor(output='eigs', doping_levels=True)
    elif target_prop.lower() == 'conductivity':
        d = self.get_conductivity(output='eigs', doping_levels=True)
    elif target_prop.lower() == 'kappa':
        d = self.get_thermal_conductivity(output='eigs', doping_levels=True)
    elif target_prop.lower() == 'zt':
        d = self.get_zt(output='eigs', doping_levels=True)
    else:
        raise ValueError(f'Unrecognized target_prop={target_prop!r}')
    abs_val = True
    x_val = x_temp = x_doping = x_isotropic = None
    output = {}
    min_temp = min_temp or 0
    max_temp = max_temp or float('inf')
    min_doping = min_doping or 0
    max_doping = max_doping or float('inf')
    for pn in ('p', 'n'):
        for t in d[pn]:
            if min_temp <= float(t) <= max_temp:
                for didx, evs in enumerate(d[pn][t]):
                    doping_lvl = self.doping[pn][didx]
                    if min_doping <= doping_lvl <= max_doping:
                        isotropic = is_isotropic(evs, isotropy_tolerance)
                        if abs_val:
                            evs = [abs(x) for x in evs]
                        val = float(sum(evs)) / len(evs) if use_average else max(evs)
                        if x_val is None or (val > x_val and maximize) or (val < x_val and (not maximize)):
                            x_val = val
                            x_temp = t
                            x_doping = doping_lvl
                            x_isotropic = isotropic
        output[pn] = {'value': x_val, 'temperature': x_temp, 'doping': x_doping, 'isotropic': x_isotropic}
        x_val = None
    if maximize:
        max_type = 'p' if output['p']['value'] >= output['n']['value'] else 'n'
    else:
        max_type = 'p' if output['p']['value'] <= output['n']['value'] else 'n'
    output['best'] = output[max_type]
    output['best']['carrier_type'] = max_type
    return output