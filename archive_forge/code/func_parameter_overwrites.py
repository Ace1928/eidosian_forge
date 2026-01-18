import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def parameter_overwrites(openmx_keyword):
    """
        In a situation conflicting ASE standard parameters and OpenMX keywords,
        ASE parameters overrides to OpenMX keywords. While doing so, units are
        converted to OpenMX unit.
        However, if both parameters and keyword are not given, we fill up that
        part in suitable manner
          openmx_keyword : key |  Name of key used in OpenMX
          keyword : value | value corresponds to openmx_keyword
          ase_parameter : key | Name of parameter used in ASE
          parameter : value | value corresponds to ase_parameter
        """
    ase_parameter = counterparts[openmx_keyword]
    keyword = parameters.get(openmx_keyword)
    parameter = parameters.get(ase_parameter)
    if parameter is not None:
        unit = standard_units.get(unit_dict.get(openmx_keyword))
        if unit is not None:
            return parameter / unit
        return parameter
    elif keyword is not None:
        return keyword
    elif 'scf' in openmx_keyword:
        return None
    else:
        return counterparts[openmx_keyword]