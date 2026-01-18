import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def getargs2(rout):
    sortargs, args = ([], rout.get('args', []))
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a]) and a not in args]
    args = auxvars + args
    if 'sortvars' in rout:
        for a in rout['sortvars']:
            if a in args:
                sortargs.append(a)
        for a in args:
            if a not in sortargs:
                sortargs.append(a)
    else:
        sortargs = auxvars + rout['args']
    return (args, sortargs)