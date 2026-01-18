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
def test_43_hash_legacy(self, use_16_legacy=False):
    """test hash() method -- legacy 'scheme' and settings keywords"""
    cc = CryptContext(**self.sample_4_dict)
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertEqual(cc.hash('password', scheme='phpass', salt='.' * 8), '$H$5........De04R5Egz0aq8Tf.1eVhY/')
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertEqual(cc.hash('password', scheme='phpass', salt='.' * 8, ident='P'), '$P$5........De04R5Egz0aq8Tf.1eVhY/')
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertEqual(cc.hash('password', rounds=1999, salt='nacl'), '$5$rounds=1999$nacl$nmfwJIxqj0csloAAvSER0B8LU0ERCAbhmMug4Twl609')
    with self.assertWarningList(['passing settings to.*is deprecated']):
        self.assertEqual(cc.hash('password', rounds=2001, salt='nacl'), '$5$rounds=2001$nacl$8PdeoPL4aXQnJ0woHhqgIw/efyfCKC2WHneOpnvF.31')
    self.assertRaises(KeyError, cc.hash, 'secret', scheme='fake')
    self.assertRaises(TypeError, cc.hash, 'secret', scheme=1)