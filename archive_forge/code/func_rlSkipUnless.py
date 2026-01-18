import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def rlSkipUnless(cond, reason, __module__=None):
    return rlSkipIf(not cond, reason, __module__=__module__)