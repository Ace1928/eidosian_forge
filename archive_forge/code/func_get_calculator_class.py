import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def get_calculator_class(name):
    """Return calculator class."""
    if name == 'asap':
        from asap3 import EMT as Calculator
    elif name == 'gpaw':
        from gpaw import GPAW as Calculator
    elif name == 'hotbit':
        from hotbit import Calculator
    elif name == 'vasp2':
        from ase.calculators.vasp import Vasp2 as Calculator
    elif name == 'ace':
        from ase.calculators.acemolecule import ACE as Calculator
    elif name == 'Psi4':
        from ase.calculators.psi4 import Psi4 as Calculator
    elif name in external_calculators:
        Calculator = external_calculators[name]
    else:
        classname = special.get(name, name.title())
        module = __import__('ase.calculators.' + name, {}, None, [classname])
        Calculator = getattr(module, classname)
    return Calculator