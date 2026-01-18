from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_classproperty(self):
    from passlib.utils.decor import classproperty

    class test(object):
        xvar = 1

        @classproperty
        def xprop(cls):
            return cls.xvar
    self.assertEqual(test.xprop, 1)
    prop = test.__dict__['xprop']
    self.assertIs(prop.im_func, prop.__func__)