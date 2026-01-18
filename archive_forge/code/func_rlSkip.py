import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def rlSkip(reason, __module__=None):
    return rlSkipIf(True, reason, __module__=__module__)