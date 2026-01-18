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
def get_interface(self, routine):
    """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
    prototype = ['interface\n']
    prototype.extend(self._get_routine_opening(routine))
    prototype.extend(self._declare_arguments(routine))
    prototype.extend(self._get_routine_ending(routine))
    prototype.append('end interface\n')
    return ''.join(prototype)