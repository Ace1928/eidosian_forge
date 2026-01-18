import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def getstringlist(self, section, option):
    """Coerce option to a list of strings or return unchanged if that fails."""
    value = ConfigParser.get(self, section, option)
    val = value.replace('\n', '')
    if self.pat.match(val):
        return eval(val, {__builtins__: None})
    else:
        return value