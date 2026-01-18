import collections
import warnings
from sympy.external import import_module
def strfunc(z):
    if z == 0:
        return ''
    elif z == 1:
        return '_d'
    else:
        return '_' + 'd' * z