import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def ischaracter_or_characterarray(var):
    return _ischaracter(var) and 'charselector' not in var