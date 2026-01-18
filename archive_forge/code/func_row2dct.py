from random import randint
from typing import Dict, Tuple, Any
import numpy as np
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import (get_calculator_class, all_properties,
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols, atomic_masses
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode
def row2dct(row, key_descriptions: Dict[str, Tuple[str, str, str]]={}) -> Dict[str, Any]:
    """Convert row to dict of things for printing or a web-page."""
    from ase.db.core import float_to_time_string, now
    dct = {}
    atoms = Atoms(cell=row.cell, pbc=row.pbc)
    dct['size'] = kptdensity2monkhorstpack(atoms, kptdensity=1.8, even=False)
    dct['cell'] = [['{:.3f}'.format(a) for a in axis] for axis in row.cell]
    par = ['{:.3f}'.format(x) for x in cell_to_cellpar(row.cell)]
    dct['lengths'] = par[:3]
    dct['angles'] = par[3:]
    stress = row.get('stress')
    if stress is not None:
        dct['stress'] = ', '.join(('{0:.3f}'.format(s) for s in stress))
    dct['formula'] = Formula(row.formula).format('abc')
    dipole = row.get('dipole')
    if dipole is not None:
        dct['dipole'] = ', '.join(('{0:.3f}'.format(d) for d in dipole))
    data = row.get('data')
    if data:
        dct['data'] = ', '.join(data.keys())
    constraints = row.get('constraints')
    if constraints:
        dct['constraints'] = ', '.join((c.__class__.__name__ for c in constraints))
    keys = {'id', 'energy', 'fmax', 'smax', 'mass', 'age'} | set(key_descriptions) | set(row.key_value_pairs)
    dct['table'] = []
    for key in keys:
        if key == 'age':
            age = float_to_time_string(now() - row.ctime, True)
            dct['table'].append(('ctime', 'Age', age))
            continue
        value = row.get(key)
        if value is not None:
            if isinstance(value, float):
                value = '{:.3f}'.format(value)
            elif not isinstance(value, str):
                value = str(value)
            desc, unit = key_descriptions.get(key, ['', '', ''])[1:]
            if unit:
                value += ' ' + unit
            dct['table'].append((key, desc, value))
    return dct