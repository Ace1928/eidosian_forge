from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_crypt(self):
    """test crypt.crypt() wrappers"""
    from passlib.utils import has_crypt, safe_crypt, test_crypt
    from passlib.registry import get_supported_os_crypt_schemes, get_crypt_handler
    supported = get_supported_os_crypt_schemes()
    if not has_crypt:
        self.assertEqual(supported, ())
        self.assertEqual(safe_crypt('test', 'aa'), None)
        self.assertFalse(test_crypt('test', 'aaqPiZY5xR5l.'))
        raise self.skipTest('crypt.crypt() not available')
    if not supported:
        raise self.fail('crypt() present, but no supported schemes found!')
    for scheme in ('md5_crypt', 'sha256_crypt'):
        if scheme in supported:
            break
    else:
        scheme = supported[-1]
    hasher = get_crypt_handler(scheme)
    if getattr(hasher, 'min_rounds', None):
        hasher = hasher.using(rounds=hasher.min_rounds)

    def get_hash(secret):
        assert isinstance(secret, unicode)
        hash = hasher.hash(secret)
        if isinstance(hash, bytes):
            hash = hash.decode('utf-8')
        assert isinstance(hash, unicode)
        return hash
    s1 = u('test')
    h1 = get_hash(s1)
    result = safe_crypt(s1, h1)
    self.assertIsInstance(result, unicode)
    self.assertEqual(result, h1)
    self.assertEqual(safe_crypt(to_bytes(s1), to_bytes(h1)), h1)
    h1x = h1[:-2] + 'xx'
    self.assertEqual(safe_crypt(s1, h1x), h1)
    s2 = u('testሴ')
    h2 = get_hash(s2)
    self.assertEqual(safe_crypt(s2, h2), h2)
    self.assertEqual(safe_crypt(to_bytes(s2), to_bytes(h2)), h2)
    self.assertRaises(ValueError, safe_crypt, '\x00', h1)
    self.assertTrue(test_crypt('test', h1))
    self.assertFalse(test_crypt('test', h1x))
    import passlib.utils as mod
    orig = mod._crypt
    try:
        retval = None
        mod._crypt = lambda secret, hash: retval
        for retval in [None, '', ':', ':0', '*0']:
            self.assertEqual(safe_crypt('test', h1), None)
            self.assertFalse(test_crypt('test', h1))
        retval = 'xxx'
        self.assertEqual(safe_crypt('test', h1), 'xxx')
        self.assertFalse(test_crypt('test', h1))
    finally:
        mod._crypt = orig