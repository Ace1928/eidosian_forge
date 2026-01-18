import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def sisitensor33(name):
    return lambda d: {'atom1': {'label': data[0], 'index': int(data[1])}, 'atom2': {'label': data[2], 'index': int(data[3])}, name: tensor33([float(x) for x in data[4:]])}