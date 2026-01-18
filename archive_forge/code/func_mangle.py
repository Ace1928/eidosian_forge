import gast as ast
import numpy as np
import numbers
def mangle(name):
    """
    Mangle a module name, except the builtins module
    >>> mangle('numpy')
    __pythran_import_numpy
    >>> mangle('builtins')
    builtins
    """
    if name == 'builtins':
        return name
    else:
        return PYTHRAN_IMPORT_MANGLING + name