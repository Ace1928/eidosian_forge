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
def dump_f95(self, routines, f, prefix, header=True, empty=True):
    for r in routines:
        lowercase = {str(x).lower() for x in r.variables}
        orig_case = {str(x) for x in r.variables}
        if len(lowercase) < len(orig_case):
            raise CodeGenError('Fortran ignores case. Got symbols: %s' % ', '.join([str(var) for var in r.variables]))
    self.dump_code(routines, f, prefix, header, empty)