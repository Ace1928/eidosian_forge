import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def iscomplexfunction_warn(rout):
    if iscomplexfunction(rout):
        outmess('    **************************************************************\n        Warning: code with a function returning complex value\n        may not work correctly with your Fortran compiler.\n        When using GNU gcc/g77 compilers, codes should work\n        correctly for callbacks with:\n        f2py -c -DF2PY_CB_RETURNCOMPLEX\n    **************************************************************\n')
        return 1
    return 0