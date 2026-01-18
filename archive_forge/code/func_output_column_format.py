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
def output_column_format(atoms, columns, arrays, write_info=True, results=None):
    """
    Helper function to build extended XYZ comment line
    """
    fmt_map = {'d': ('R', '%16.8f'), 'f': ('R', '%16.8f'), 'i': ('I', '%8d'), 'O': ('S', '%s'), 'S': ('S', '%s'), 'U': ('S', '%-2s'), 'b': ('L', ' %.1s')}
    lattice_str = 'Lattice="' + ' '.join([str(x) for x in np.reshape(atoms.cell.T, 9, order='F')]) + '"'
    property_names = []
    property_types = []
    property_ncols = []
    dtypes = []
    formats = []
    for column in columns:
        array = arrays[column]
        dtype = array.dtype
        property_name = PROPERTY_NAME_MAP.get(column, column)
        property_type, fmt = fmt_map[dtype.kind]
        property_names.append(property_name)
        property_types.append(property_type)
        if len(array.shape) == 1 or (len(array.shape) == 2 and array.shape[1] == 1):
            ncol = 1
            dtypes.append((column, dtype))
        else:
            ncol = array.shape[1]
            for c in range(ncol):
                dtypes.append((column + str(c), dtype))
        formats.extend([fmt] * ncol)
        property_ncols.append(ncol)
    props_str = ':'.join([':'.join(x) for x in zip(property_names, property_types, [str(nc) for nc in property_ncols])])
    comment_str = ''
    if atoms.cell.any():
        comment_str += lattice_str + ' '
    comment_str += 'Properties={}'.format(props_str)
    info = {}
    if write_info:
        info.update(atoms.info)
    if results is not None:
        info.update(results)
    info['pbc'] = atoms.get_pbc()
    comment_str += ' ' + key_val_dict_to_str(info)
    dtype = np.dtype(dtypes)
    fmt = ' '.join(formats) + '\n'
    return (comment_str, property_ncols, dtype, fmt)