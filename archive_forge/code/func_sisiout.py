import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def sisiout(tag, tensor_name):
    if tag in data:
        for isc in data[tag]:
            out.append('  %s %s %d %s %d %s' % (tag, isc['atom1']['label'], isc['atom1']['index'], isc['atom2']['label'], isc['atom2']['index'], tensor_string(isc[tensor_name])))