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
def removespaces(expr):
    expr = expr.strip()
    if len(expr) <= 1:
        return expr
    expr2 = expr[0]
    for i in range(1, len(expr) - 1):
        if expr[i] == ' ' and (expr[i + 1] in '()[]{}=+-/* ' or expr[i - 1] in '()[]{}=+-/* '):
            continue
        expr2 = expr2 + expr[i]
    expr2 = expr2 + expr[-1]
    return expr2