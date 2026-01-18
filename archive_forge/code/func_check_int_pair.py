from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def check_int_pair(self, bits, encoded_pairs):
    """helper to check encode_intXX & decode_intXX functions"""
    rng = self.getRandom()
    engine = self.engine
    encode = getattr(engine, 'encode_int%s' % bits)
    decode = getattr(engine, 'decode_int%s' % bits)
    pad = -bits % 6
    chars = (bits + pad) // 6
    upper = 1 << bits
    for value, encoded in encoded_pairs:
        result = encode(value)
        self.assertIsInstance(result, bytes)
        self.assertEqual(result, encoded)
    self.assertRaises(ValueError, encode, -1)
    self.assertRaises(ValueError, encode, upper)
    for value, encoded in encoded_pairs:
        self.assertEqual(decode(encoded), value, 'encoded %r:' % (encoded,))
    m = self.m
    self.assertRaises(ValueError, decode, m(0) * (chars + 1))
    self.assertRaises(ValueError, decode, m(0) * (chars - 1))
    self.assertRaises(ValueError, decode, self.bad_byte * chars)
    self.assertRaises(TypeError, decode, engine.charmap[0])
    self.assertRaises(TypeError, decode, None)
    from passlib.utils import getrandstr
    for i in irange(100):
        value = rng.randint(0, upper - 1)
        encoded = encode(value)
        self.assertEqual(len(encoded), chars)
        self.assertEqual(decode(encoded), value)
        encoded = getrandstr(rng, engine.bytemap, chars)
        value = decode(encoded)
        self.assertGreaterEqual(value, 0, 'decode %r out of bounds:' % encoded)
        self.assertLess(value, upper, 'decode %r out of bounds:' % encoded)
        result = encode(value)
        if pad:
            self.assertEqual(result[:-2], encoded[:-2])
        else:
            self.assertEqual(result, encoded)