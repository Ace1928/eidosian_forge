from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_genseed(self):
    """test genseed()"""
    import random
    from passlib.utils import genseed
    rng = random.Random(genseed())
    a = rng.randint(0, 10 ** 10)
    rng = random.Random(genseed())
    b = rng.randint(0, 10 ** 10)
    self.assertNotEqual(a, b)
    rng.seed(genseed(rng))