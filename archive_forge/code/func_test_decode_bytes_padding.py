from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_decode_bytes_padding(self):
    """test decode_bytes() ignores padding bits"""
    bchr = (lambda v: bytes([v])) if PY3 else chr
    engine = self.engine
    m = self.m
    decode = engine.decode_bytes
    BNULL = b'\x00'
    self.assertEqual(decode(m(0, 0)), BNULL)
    for i in range(0, 6):
        if engine.big:
            correct = BNULL if i < 4 else bchr(1 << i - 4)
        else:
            correct = bchr(1 << i + 6) if i < 2 else BNULL
        self.assertEqual(decode(m(0, 1 << i)), correct, '%d/4 bits:' % i)
    self.assertEqual(decode(m(0, 0, 0)), BNULL * 2)
    for i in range(0, 6):
        if engine.big:
            correct = BNULL if i < 2 else bchr(1 << i - 2)
        else:
            correct = bchr(1 << i + 4) if i < 4 else BNULL
        self.assertEqual(decode(m(0, 0, 1 << i)), BNULL + correct, '%d/2 bits:' % i)