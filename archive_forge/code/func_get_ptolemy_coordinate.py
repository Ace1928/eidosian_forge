from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def get_ptolemy_coordinate(addl_index):
    total_index = matrix.vector_add(index, addl_index)
    key = 'c_%d%d%d%d' % tuple(total_index) + '_%d' % tet
    return self[key]