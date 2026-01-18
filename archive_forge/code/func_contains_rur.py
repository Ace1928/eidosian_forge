import re
from .utilities import MethodMappingList
from .component import NonZeroDimensionalComponent
from .coordinates import PtolemyCoordinates
from .rur import RUR
from . import processFileBase
from ..pari import pari
def contains_rur(text):
    return 'RUR=DECOMPOSITION=BEGINS' in text