from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_consteq(self):
    """test consteq()"""
    from passlib.utils import consteq, str_consteq
    self.assertRaises(TypeError, consteq, u(''), b'')
    self.assertRaises(TypeError, consteq, u(''), 1)
    self.assertRaises(TypeError, consteq, u(''), None)
    self.assertRaises(TypeError, consteq, b'', u(''))
    self.assertRaises(TypeError, consteq, b'', 1)
    self.assertRaises(TypeError, consteq, b'', None)
    self.assertRaises(TypeError, consteq, None, u(''))
    self.assertRaises(TypeError, consteq, None, b'')
    self.assertRaises(TypeError, consteq, 1, u(''))
    self.assertRaises(TypeError, consteq, 1, b'')

    def consteq_supports_string(value):
        return consteq is str_consteq or PY2 or is_ascii_safe(value)
    for value in [u('a'), u('abc'), u('ÿ¢\x12\x00') * 10]:
        if consteq_supports_string(value):
            self.assertTrue(consteq(value, value), 'value %r:' % (value,))
        else:
            self.assertRaises(TypeError, consteq, value, value)
        self.assertTrue(str_consteq(value, value), 'value %r:' % (value,))
        value = value.encode('latin-1')
        self.assertTrue(consteq(value, value), 'value %r:' % (value,))
    for l, r in [(u('a'), u('c')), (u('abcabc'), u('zbaabc')), (u('abcabc'), u('abzabc')), (u('abcabc'), u('abcabz')), ((u('ÿ¢\x12\x00') * 10)[:-1] + u('\x01'), u('ÿ¢\x12\x00') * 10), (u(''), u('a')), (u('abc'), u('abcdef')), (u('abc'), u('defabc')), (u('qwertyuiopasdfghjklzxcvbnm'), u('abc'))]:
        if consteq_supports_string(l) and consteq_supports_string(r):
            self.assertFalse(consteq(l, r), 'values %r %r:' % (l, r))
            self.assertFalse(consteq(r, l), 'values %r %r:' % (r, l))
        else:
            self.assertRaises(TypeError, consteq, l, r)
            self.assertRaises(TypeError, consteq, r, l)
        self.assertFalse(str_consteq(l, r), 'values %r %r:' % (l, r))
        self.assertFalse(str_consteq(r, l), 'values %r %r:' % (r, l))
        l = l.encode('latin-1')
        r = r.encode('latin-1')
        self.assertFalse(consteq(l, r), 'values %r %r:' % (l, r))
        self.assertFalse(consteq(r, l), 'values %r %r:' % (r, l))