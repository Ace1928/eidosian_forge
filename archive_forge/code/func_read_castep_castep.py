import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_castep_castep(fd, index=None):
    """
    Reads a .castep file and returns an atoms  object.
    The calculator information will be stored in the calc attribute.

    There is no use of the "index" argument as of now, it is just inserted for
    convenience to comply with the generic "read()" in ase.io

    Please note that this routine will return an atom ordering as found
    within the castep file. This means that the species will be ordered by
    ascending atomic numbers. The atoms witin a species are ordered as given
    in the original cell file.

    Note: This routine returns a single atoms_object only, the last
    configuration in the file. Yet, if you want to parse an MD run, use the
    novel function `read_md()`
    """
    from ase.calculators.castep import Castep
    try:
        calc = Castep()
    except Exception as e:
        warnings.warn('WARNING: {0} Using fallback .castep reader...'.format(e))
        return read_castep_castep_old(fd, index)
    calc.read(castep_file=fd)
    calc._old_atoms = calc.atoms
    calc._old_param = calc.param
    calc._old_cell = calc.cell
    return [calc.atoms]