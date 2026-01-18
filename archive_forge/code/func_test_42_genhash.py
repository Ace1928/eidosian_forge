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
def test_42_genhash(self):
    """test genhash() method"""
    cc = CryptContext(['des_crypt'])
    hash = cc.hash('stub')
    for secret, kwds in self.nonstring_vectors:
        self.assertRaises(TypeError, cc.genhash, secret, hash, **kwds)
    cc = CryptContext(['des_crypt'])
    for config, kwds in self.nonstring_vectors:
        if hash is None:
            continue
        self.assertRaises(TypeError, cc.genhash, 'secret', config, **kwds)
    cc = CryptContext(['mysql323'])
    self.assertRaises(TypeError, cc.genhash, 'stub', None)
    self.assertRaises(KeyError, CryptContext().genhash, 'secret', 'hash')
    self.assertRaises(KeyError, cc.genhash, 'secret', hash, scheme='fake')
    self.assertRaises(TypeError, cc.genhash, 'secret', hash, scheme=1)
    self.assertRaises(TypeError, cc.genconfig, 'secret', hash, category=1)