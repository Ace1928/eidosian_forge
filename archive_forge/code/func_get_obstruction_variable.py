from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def get_obstruction_variable(face):
    key = 's_%d_%d' % (face, tet)
    return solution_dict[key]