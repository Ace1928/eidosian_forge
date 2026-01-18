import datetime
import os
import re
import sys
from collections import OrderedDict
import numpy
from . import units
from .colormap import ColorMap
from .Point import Point
from .Qt import QtCore
def measureIndent(s):
    n = 0
    while n < len(s) and s[n] == ' ':
        n += 1
    return n