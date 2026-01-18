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
def read_pnm_header(infile, supported=('P5', 'P6')):
    """
    Read a PNM header, returning (format,width,height,depth,maxval).
    `width` and `height` are in pixels.  `depth` is the number of
    channels in the image; for PBM and PGM it is synthesized as 1, for
    PPM as 3; for PAM images it is read from the header.  `maxval` is
    synthesized (as 1) for PBM images.
    """
    supported = [strtobytes(x) for x in supported]
    type = infile.read(3).rstrip()
    if type not in supported:
        raise NotImplementedError(f'file format {type} not supported')
    if type == strtobytes('P7'):
        return read_pam_header(infile)
    expected = 4
    pbm = ('P1', 'P4')
    if type in pbm:
        expected = 3
    header = [type]

    def getc():
        c = infile.read(1)
        if not c:
            raise Error('premature EOF reading PNM header')
        return c
    c = getc()
    while True:
        while c.isspace():
            c = getc()
        while c == '#':
            while c not in '\n\r':
                c = getc()
        if not c.isdigit():
            raise Error(f'unexpected character {c} found in header')
        token = strtobytes('')
        while c.isdigit():
            token += c
            c = getc()
        header.append(int(token))
        if len(header) == expected:
            break
    while c == '#':
        while c not in '\n\r':
            c = getc()
    if not c.isspace():
        raise Error(f'expected header to end with whitespace, not {c}')
    if type in pbm:
        header.append(1)
    depth = (1, 3)[type == strtobytes('P6')]
    return (header[0], header[1], header[2], depth, header[3])