import re
import warnings
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry
def exportmany(prototypes):
    warnings.warn('exportmany() is deprecated, use the numba.pycc.CC API instead', DeprecationWarning, stacklevel=2)

    def wrapped(func):
        for proto in prototypes:
            export(proto)(func)
    return wrapped