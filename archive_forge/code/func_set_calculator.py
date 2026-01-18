import sys
from typing import Dict, Any
import numpy as np
from ase.calculators.calculator import (get_calculator_class,
from ase.constraints import FixAtoms, UnitCellFilter
from ase.eos import EquationOfState
from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
import ase.db as db
def set_calculator(self, atoms, name):
    cls = get_calculator_class(self.calculator_name)
    parameters = str2dict(self.args.parameters)
    if getattr(cls, 'nolabel', False):
        atoms.calc = cls(**parameters)
    else:
        atoms.calc = cls(label=self.get_filename(name), **parameters)