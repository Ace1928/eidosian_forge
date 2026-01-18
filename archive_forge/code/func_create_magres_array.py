import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def create_magres_array(name, order, block):
    if order == 1:
        u_arr = [None] * len(li_list)
    elif order == 2:
        u_arr = [[None] * (i + 1) for i in range(len(li_list))]
    else:
        raise ValueError('Invalid order value passed to create_magres_array')
    for s in block:
        if order == 1:
            at = (s['atom']['label'], s['atom']['index'])
            try:
                ai = li_list.index(at)
            except ValueError:
                raise RuntimeError('Invalid data in magres block')
            u_arr[ai] = s[mn]
        else:
            at1 = (s['atom1']['label'], s['atom1']['index'])
            at2 = (s['atom2']['label'], s['atom2']['index'])
            ai1 = li_list.index(at1)
            ai2 = li_list.index(at2)
            ai1, ai2 = sorted((ai1, ai2), reverse=True)
            u_arr[ai1][ai2] = s[mn]
    if order == 1:
        return np.array(u_arr)
    else:
        return np.array(u_arr, dtype=object)