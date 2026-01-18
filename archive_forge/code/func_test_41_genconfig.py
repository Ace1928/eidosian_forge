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
def test_41_genconfig(self):
    """test genconfig() method"""
    cc = CryptContext(schemes=['md5_crypt', 'phpass'], phpass__ident='H', phpass__default_rounds=7, admin__phpass__ident='P')
    self.assertTrue(cc.genconfig().startswith('$1$'))
    self.assertTrue(cc.genconfig(scheme='phpass').startswith('$H$5'))
    self.assertTrue(cc.genconfig(scheme='phpass', category='admin').startswith('$P$5'))
    self.assertTrue(cc.genconfig(scheme='phpass', category='staff').startswith('$H$5'))
    self.assertEqual(cc.genconfig(scheme='phpass', salt='.' * 8, rounds=8, ident='P'), '$P$6........22zGEuacuPOqEpYPDeR0R/')
    if PY2:
        c2 = cc.copy(default='phpass')
        self.assertTrue(c2.genconfig(category=u('admin')).startswith('$P$5'))
        self.assertTrue(c2.genconfig(category=u('staff')).startswith('$H$5'))
    self.assertRaises(KeyError, CryptContext().genconfig)
    self.assertRaises(KeyError, CryptContext().genconfig, scheme='md5_crypt')
    self.assertRaises(KeyError, cc.genconfig, scheme='fake')
    self.assertRaises(TypeError, cc.genconfig, scheme=1, category='staff')
    self.assertRaises(TypeError, cc.genconfig, scheme=1)
    self.assertRaises(TypeError, cc.genconfig, category=1)