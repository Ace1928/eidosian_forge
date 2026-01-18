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
def atoms2dict(atoms):
    dct = {'numbers': atoms.numbers, 'positions': atoms.positions, 'unique_id': '%x' % randint(16 ** 31, 16 ** 32 - 1)}
    if atoms.pbc.any():
        dct['pbc'] = atoms.pbc
    if atoms.cell.any():
        dct['cell'] = atoms.cell
    if atoms.has('initial_magmoms'):
        dct['initial_magmoms'] = atoms.get_initial_magnetic_moments()
    if atoms.has('initial_charges'):
        dct['initial_charges'] = atoms.get_initial_charges()
    if atoms.has('masses'):
        dct['masses'] = atoms.get_masses()
    if atoms.has('tags'):
        dct['tags'] = atoms.get_tags()
    if atoms.has('momenta'):
        dct['momenta'] = atoms.get_momenta()
    if atoms.constraints:
        dct['constraints'] = [c.todict() for c in atoms.constraints]
    if atoms.calc is not None:
        dct['calculator'] = atoms.calc.name.lower()
        dct['calculator_parameters'] = atoms.calc.todict()
        if len(atoms.calc.check_state(atoms)) == 0:
            for prop in all_properties:
                try:
                    x = atoms.calc.get_property(prop, atoms, False)
                except PropertyNotImplementedError:
                    pass
                else:
                    if x is not None:
                        dct[prop] = x
    return dct