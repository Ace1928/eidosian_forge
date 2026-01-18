import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def eqCheck(r, x):
    if r != x:
        print('Strings unequal\nexp: %s\ngot: %s' % (ascii(x), ascii(r)))