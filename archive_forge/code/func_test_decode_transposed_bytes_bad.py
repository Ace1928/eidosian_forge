from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_decode_transposed_bytes_bad(self):
    """test decode_transposed_bytes() fails if map is a one-way"""
    engine = self.engine
    for input, _, offsets in self.transposed_dups:
        tmp = engine.encode_bytes(input)
        self.assertRaises(TypeError, engine.decode_transposed_bytes, tmp, offsets)