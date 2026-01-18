from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_codec(self):
    """test encode_bytes/decode_bytes against random data"""
    engine = self.engine
    from passlib.utils import getrandbytes, getrandstr
    rng = self.getRandom()
    saw_zero = False
    for i in irange(500):
        size = rng.randint(1 if saw_zero else 0, 12)
        if not size:
            saw_zero = True
        enc_size = (4 * size + 2) // 3
        raw = getrandbytes(rng, size)
        encoded = engine.encode_bytes(raw)
        self.assertEqual(len(encoded), enc_size)
        result = engine.decode_bytes(encoded)
        self.assertEqual(result, raw)
        if size % 4 == 1:
            size += rng.choice([-1, 1, 2])
        raw_size = 3 * size // 4
        encoded = getrandstr(rng, engine.bytemap, size)
        raw = engine.decode_bytes(encoded)
        self.assertEqual(len(raw), raw_size, 'encoded %d:' % size)
        result = engine.encode_bytes(raw)
        if size % 4:
            self.assertEqual(result[:-1], encoded[:-1])
        else:
            self.assertEqual(result, encoded)