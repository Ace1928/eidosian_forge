import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
@classmethod
def from_fortran(cls, fmt):
    if not len(fmt) == 3:
        raise ValueError('Fortran format for matrix type should be 3 characters long')
    try:
        value_type = cls._f2q_type[fmt[0]]
        structure = cls._f2q_structure[fmt[1]]
        storage = cls._f2q_storage[fmt[2]]
        return cls(value_type, structure, storage)
    except KeyError as e:
        raise ValueError('Unrecognized format %s' % fmt) from e