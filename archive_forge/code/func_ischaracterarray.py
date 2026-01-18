import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def ischaracterarray(var):
    return ischaracter_or_characterarray(var) and isarray(var)