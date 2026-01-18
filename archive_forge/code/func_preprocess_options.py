from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
def preprocess_options(args):
    for option, value in args.items():
        try:
            cls = self.__options__[option]
        except KeyError:
            raise OptionError("'%s' is not a valid option" % option)
        if issubclass(cls, Flag):
            if flags is None or option not in flags:
                if strict:
                    raise OptionError("'%s' flag is not allowed in this context" % option)
        if value is not None:
            self[option] = cls.preprocess(value)