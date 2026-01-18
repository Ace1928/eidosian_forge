import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def l_and(*f):
    l1, l2 = ('lambda v', [])
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % i)
    return eval('%s:%s' % (l1, ' and '.join(l2)))