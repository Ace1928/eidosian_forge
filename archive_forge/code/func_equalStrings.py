import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def equalStrings(a, b, enc='utf8'):
    return a == b if type(a) == type(b) else asUnicode(a, enc) == asUnicode(b, enc)