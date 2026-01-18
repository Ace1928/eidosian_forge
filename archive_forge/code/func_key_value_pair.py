from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def key_value_pair(v, t, index):
    new_key = v + '_%d%d%d%d' % tuple(index) + '_%d' % t
    old_key = v + '_0000' + '_%d' % t
    return (new_key, self[old_key])