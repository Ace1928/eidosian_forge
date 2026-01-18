import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def proplist(self, rec=0):
    """ return a mapping of property names to property values.
If rec is True, then return a dictionary mapping sub-paths to such mappings.
"""
    if rec:
        res = self._svn('proplist -R')
        return make_recursive_propdict(self, res)
    else:
        res = self._svn('proplist')
        lines = res.split('\n')
        lines = [x.strip() for x in lines[1:]]
        return PropListDict(self, lines)