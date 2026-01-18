from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def get_N(key):
    m = re.match('c_(\\d{4})_\\d+$', key)
    if not m:
        raise Exception("Not a valid Ptolemy key: '%s'" % key)
    return sum([int(char) for char in m.group(1)])