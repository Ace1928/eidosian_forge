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
def get_code_generator(language, project=None, standard=None, printer=None):
    if language == 'C':
        if standard is None:
            pass
        elif standard.lower() == 'c89':
            language = 'C89'
        elif standard.lower() == 'c99':
            language = 'C99'
    CodeGenClass = {'C': CCodeGen, 'C89': C89CodeGen, 'C99': C99CodeGen, 'F95': FCodeGen, 'JULIA': JuliaCodeGen, 'OCTAVE': OctaveCodeGen, 'RUST': RustCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project, printer)