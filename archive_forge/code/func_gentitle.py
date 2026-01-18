import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def gentitle(name):
    ln = (80 - len(name) - 6) // 2
    return '/*%s %s %s*/' % (ln * '*', name, ln * '*')