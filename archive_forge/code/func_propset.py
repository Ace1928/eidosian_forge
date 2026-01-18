import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def propset(self, name, value, *args):
    """ set property name to value on this path. """
    d = py.path.local.mkdtemp()
    try:
        p = d.join('value')
        p.write(value)
        self._svn('propset', name, '--file', str(p), *args)
    finally:
        d.remove()