import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def header_alias(h):
    """Replace keyboard characters with Unicode symbols
    for pretty printing"""
    if h == 'i':
        h = 'index'
    elif h == 'an':
        h = 'atomic #'
    elif h == 't':
        h = 'tag'
    elif h == 'el':
        h = 'element'
    elif h[0] == 'd':
        h = h.replace('d', 'Δ')
    elif h[0] == 'r':
        h = 'rank ' + header_alias(h[1:])
    elif h[0] == 'a':
        h = h.replace('a', '<')
        h += '>'
    return h