import sys
from typing import Dict, Any
import numpy as np
from ase.calculators.calculator import (get_calculator_class,
from ase.constraints import FixAtoms, UnitCellFilter
from ase.eos import EquationOfState
from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
import ase.db as db
def str2dict(s: str, namespace={}, sep: str='=') -> Dict[str, Any]:
    """Convert comma-separated key=value string to dictionary.

    Examples:

    >>> str2dict('xc=PBE,nbands=200,parallel={band:4}')
    {'xc': 'PBE', 'nbands': 200, 'parallel': {'band': 4}}
    >>> str2dict('a=1.2,b=True,c=ab,d=1,2,3,e={f:42,g:cd}')
    {'a': 1.2, 'c': 'ab', 'b': True, 'e': {'g': 'cd', 'f': 42}, 'd': (1, 2, 3)}
    """

    def myeval(value):
        try:
            value = eval(value, namespace)
        except (NameError, SyntaxError):
            pass
        return value
    dct = {}
    strings = (s + ',').split(sep)
    for i in range(len(strings) - 1):
        key = strings[i]
        m = strings[i + 1].rfind(',')
        value: Any = strings[i + 1][:m]
        if value[0] == '{':
            assert value[-1] == '}'
            value = str2dict(value[1:-1], namespace, ':')
        elif value[0] == '(':
            assert value[-1] == ')'
            value = [myeval(t) for t in value[1:-1].split(',')]
        else:
            value = myeval(value)
        dct[key] = value
        strings[i + 1] = strings[i + 1][m + 1:]
    return dct