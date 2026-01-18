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
def cracktypespec0(typespec, ll):
    selector = None
    attr = None
    if re.match('double\\s*complex', typespec, re.I):
        typespec = 'double complex'
    elif re.match('double\\s*precision', typespec, re.I):
        typespec = 'double precision'
    else:
        typespec = typespec.strip().lower()
    m1 = selectpattern.match(markouterparen(ll))
    if not m1:
        outmess('cracktypespec0: no kind/char_selector pattern found for line.\n')
        return
    d = m1.groupdict()
    for k in list(d.keys()):
        d[k] = unmarkouterparen(d[k])
    if typespec in ['complex', 'integer', 'logical', 'real', 'character', 'type']:
        selector = d['this']
        ll = d['after']
    i = ll.find('::')
    if i >= 0:
        attr = ll[:i].strip()
        ll = ll[i + 2:]
    return (typespec, selector, attr, ll)