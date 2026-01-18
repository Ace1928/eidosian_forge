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
def test_40_basic(self):
    """test basic hash/identify/verify functionality"""
    handlers = [hash.md5_crypt, hash.des_crypt, hash.bsdi_crypt]
    cc = CryptContext(handlers, bsdi_crypt__default_rounds=5)
    for crypt in handlers:
        h = cc.hash('test', scheme=crypt.name)
        self.assertEqual(cc.identify(h), crypt.name)
        self.assertEqual(cc.identify(h, resolve=True, unconfigured=True), crypt)
        self.assertHandlerDerivedFrom(cc.identify(h, resolve=True), crypt)
        self.assertTrue(cc.verify('test', h))
        self.assertFalse(cc.verify('notest', h))
    h = cc.hash('test')
    self.assertEqual(cc.identify(h), 'md5_crypt')
    h = cc.genhash('secret', cc.genconfig())
    self.assertEqual(cc.identify(h), 'md5_crypt')
    h = cc.genhash('secret', cc.genconfig(), scheme='md5_crypt')
    self.assertEqual(cc.identify(h), 'md5_crypt')
    self.assertRaises(ValueError, cc.genhash, 'secret', cc.genconfig(), scheme='des_crypt')