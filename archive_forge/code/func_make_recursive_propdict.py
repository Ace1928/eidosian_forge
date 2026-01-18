import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def make_recursive_propdict(wcroot, output, rex=re.compile("Properties on '(.*)':")):
    """ Return a dictionary of path->PropListDict mappings. """
    lines = [x for x in output.split('\n') if x]
    pdict = {}
    while lines:
        line = lines.pop(0)
        m = rex.match(line)
        if not m:
            raise ValueError('could not parse propget-line: %r' % line)
        path = m.groups()[0]
        wcpath = wcroot.join(path, abs=1)
        propnames = []
        while lines and lines[0].startswith('  '):
            propname = lines.pop(0).strip()
            propnames.append(propname)
        assert propnames, 'must have found properties!'
        pdict[wcpath] = PropListDict(wcpath, propnames)
    return pdict