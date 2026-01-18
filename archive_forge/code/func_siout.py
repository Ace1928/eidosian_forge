import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def siout(tag, tensor_name):
    if tag in data:
        for atom_si in data[tag]:
            out.append('  %s %s %d %s' % (tag, atom_si['atom']['label'], atom_si['atom']['index'], tensor_string(atom_si[tensor_name])))