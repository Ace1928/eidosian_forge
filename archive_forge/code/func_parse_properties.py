from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
def parse_properties(prop_str):
    """
    Parse extended XYZ properties format string

    Format is "[NAME:TYPE:NCOLS]...]", e.g. "species:S:1:pos:R:3".
    NAME is the name of the property.
    TYPE is one of R, I, S, L for real, integer, string and logical.
    NCOLS is number of columns for that property.
    """
    properties = {}
    properties_list = []
    dtypes = []
    converters = []
    fields = prop_str.split(':')

    def parse_bool(x):
        """
        Parse bool to string
        """
        return {'T': True, 'F': False, 'True': True, 'False': False}.get(x)
    fmt_map = {'R': ('d', float), 'I': ('i', int), 'S': (object, str), 'L': ('bool', parse_bool)}
    for name, ptype, cols in zip(fields[::3], fields[1::3], [int(x) for x in fields[2::3]]):
        if ptype not in ('R', 'I', 'S', 'L'):
            raise ValueError('Unknown property type: ' + ptype)
        ase_name = REV_PROPERTY_NAME_MAP.get(name, name)
        dtype, converter = fmt_map[ptype]
        if cols == 1:
            dtypes.append((name, dtype))
            converters.append(converter)
        else:
            for c in range(cols):
                dtypes.append((name + str(c), dtype))
                converters.append(converter)
        properties[name] = (ase_name, cols)
        properties_list.append(name)
    dtype = np.dtype(dtypes)
    return (properties, properties_list, dtype, converters)