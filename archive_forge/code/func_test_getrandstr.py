from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
@run_with_fixed_seeds(count=1024)
def test_getrandstr(self, seed):
    """getrandstr()"""
    from passlib.utils import getrandstr
    wrapper = partial(getrandstr, self.getRandom(seed=seed))
    self.assertEqual(wrapper('abc', 0), '')
    self.assertRaises(ValueError, wrapper, 'abc', -1)
    self.assertRaises(ValueError, wrapper, '', 0)
    self.assertEqual(wrapper('a', 5), 'aaaaa')
    x = wrapper(u('abc'), 32)
    y = wrapper(u('abc'), 32)
    self.assertIsInstance(x, unicode)
    self.assertNotEqual(x, y)
    self.assertEqual(sorted(set(x)), [u('a'), u('b'), u('c')])
    x = wrapper(b'abc', 32)
    y = wrapper(b'abc', 32)
    self.assertIsInstance(x, bytes)
    self.assertNotEqual(x, y)
    self.assertEqual(sorted(set(x.decode('ascii'))), [u('a'), u('b'), u('c')])