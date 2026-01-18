from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
def build_options(gens, args=None):
    """Construct options from keyword arguments or ... options. """
    if args is None:
        gens, args = ((), gens)
    if len(args) != 1 or 'opt' not in args or gens:
        return Options(gens, args)
    else:
        return args['opt']