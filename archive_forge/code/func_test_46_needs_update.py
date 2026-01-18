from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_46_needs_update(self):
    """test needs_update() method"""
    cc = CryptContext(**self.sample_4_dict)
    self.assertTrue(cc.needs_update('9XXD4trGYeGJA'))
    self.assertFalse(cc.needs_update('$1$J8HC2RCr$HcmM.7NxB2weSvlw2FgzU0'))
    self.assertTrue(cc.needs_update('$5$rounds=1999$jD81UCoo.zI.UETs$Y7qSTQ6mTiU9qZB4fRr43wRgQq4V.5AAf7F97Pzxey/'))
    self.assertFalse(cc.needs_update('$5$rounds=2000$228SSRje04cnNCaQ$YGV4RYu.5sNiBvorQDlO0WWQjyJVGKBcJXz3OtyQ2u8'))
    self.assertFalse(cc.needs_update('$5$rounds=3000$fS9iazEwTKi7QPW4$VasgBC8FqlOvD7x2HhABaMXCTh9jwHclPA9j5YQdns.'))
    self.assertTrue(cc.needs_update('$5$rounds=3001$QlFHHifXvpFX4PLs$/0ekt7lSs/lOikSerQ0M/1porEHxYq7W/2hdFpxA3fA'))
    check_state = []

    class dummy(uh.StaticHandler):
        name = 'dummy'
        _hash_prefix = '@'

        @classmethod
        def needs_update(cls, hash, secret=None):
            check_state.append((hash, secret))
            return secret == 'nu'

        def _calc_checksum(self, secret):
            from hashlib import md5
            if isinstance(secret, unicode):
                secret = secret.encode('utf-8')
            return str_to_uascii(md5(secret).hexdigest())
    ctx = CryptContext([dummy])
    hash = refhash = dummy.hash('test')
    self.assertFalse(ctx.needs_update(hash))
    self.assertEqual(check_state, [(hash, None)])
    del check_state[:]
    self.assertFalse(ctx.needs_update(hash, secret='bob'))
    self.assertEqual(check_state, [(hash, 'bob')])
    del check_state[:]
    self.assertTrue(ctx.needs_update(hash, secret='nu'))
    self.assertEqual(check_state, [(hash, 'nu')])
    del check_state[:]
    cc = CryptContext(['des_crypt'])
    for hash, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.needs_update, hash, **kwds)
    self.assertRaises(KeyError, CryptContext().needs_update, 'hash')
    self.assertRaises(KeyError, cc.needs_update, refhash, scheme='fake')
    self.assertRaises(TypeError, cc.needs_update, refhash, scheme=1)
    self.assertRaises(TypeError, cc.needs_update, refhash, category=1)