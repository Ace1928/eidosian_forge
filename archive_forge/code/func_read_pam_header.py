from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def read_pam_header(infile):
    """
    Read (the rest of a) PAM header.  `infile` should be positioned
    immediately after the initial 'P7' line (at the beginning of the
    second line).  Returns are as for `read_pnm_header`.
    """
    header = dict()
    while True:
        l = infile.readline().strip()
        if l == strtobytes('ENDHDR'):
            break
        if not l:
            raise EOFError('PAM ended prematurely')
        if l[0] == strtobytes('#'):
            continue
        l = l.split(None, 1)
        if l[0] not in header:
            header[l[0]] = l[1]
        else:
            header[l[0]] += strtobytes(' ') + l[1]
    required = ['WIDTH', 'HEIGHT', 'DEPTH', 'MAXVAL']
    required = [strtobytes(x) for x in required]
    WIDTH, HEIGHT, DEPTH, MAXVAL = required
    present = [x for x in required if x in header]
    if len(present) != len(required):
        raise Error('PAM file must specify WIDTH, HEIGHT, DEPTH, and MAXVAL')
    width = int(header[WIDTH])
    height = int(header[HEIGHT])
    depth = int(header[DEPTH])
    maxval = int(header[MAXVAL])
    if width <= 0 or height <= 0 or depth <= 0 or (maxval <= 0):
        raise Error('WIDTH, HEIGHT, DEPTH, MAXVAL must all be positive integers')
    return ('P7', width, height, depth, maxval)