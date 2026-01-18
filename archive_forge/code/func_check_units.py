import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def check_units(d):
    """
            Verify that given units for a particular tag are correct.
        """
    allowed_units = {'lattice': 'Angstrom', 'atom': 'Angstrom', 'ms': 'ppm', 'efg': 'au', 'efg_local': 'au', 'efg_nonlocal': 'au', 'isc': '10^19.T^2.J^-1', 'isc_fc': '10^19.T^2.J^-1', 'isc_orbital_p': '10^19.T^2.J^-1', 'isc_orbital_d': '10^19.T^2.J^-1', 'isc_spin': '10^19.T^2.J^-1', 'isc': '10^19.T^2.J^-1', 'sus': '10^-6.cm^3.mol^-1', 'calc_cutoffenergy': 'Hartree'}
    if d[0] in d and d[1] == allowed_units[d[0]]:
        pass
    else:
        raise RuntimeError('Unrecognized units: %s %s' % (d[0], d[1]))
    return d