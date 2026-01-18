import os
import textwrap
from io import StringIO
from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
from sympy.utilities.iterables import is_sequence
def get_default_datatype(expr, complex_allowed=None):
    """Derives an appropriate datatype based on the expression."""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = 'complex'
    else:
        final_dtype = 'float'
    if expr.is_integer:
        return default_datatypes['int']
    elif expr.is_real:
        return default_datatypes['float']
    elif isinstance(expr, MatrixBase):
        dt = 'int'
        for element in expr:
            if dt == 'int' and (not element.is_integer):
                dt = 'float'
            if dt == 'float' and (not element.is_real):
                return default_datatypes[final_dtype]
        return default_datatypes[dt]
    else:
        return default_datatypes[final_dtype]