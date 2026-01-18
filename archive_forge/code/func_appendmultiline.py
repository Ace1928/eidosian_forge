import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def appendmultiline(group, context_name, ml):
    if 'f2pymultilines' not in group:
        group['f2pymultilines'] = {}
    d = group['f2pymultilines']
    if context_name not in d:
        d[context_name] = []
    d[context_name].append(ml)
    return