from __future__ import with_statement
from logging import getLogger
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, CryptPolicy, LazyCryptContext
from passlib.utils import to_bytes, to_unicode
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase, set_file
from passlib.registry import (register_crypt_handler_path,
def test_00_constructor(self):
    """test constructor"""
    cc = CryptContext([hash.md5_crypt, hash.bsdi_crypt, hash.des_crypt])
    c, b, a = cc.policy.iter_handlers()
    self.assertIs(a, hash.des_crypt)
    self.assertIs(b, hash.bsdi_crypt)
    self.assertIs(c, hash.md5_crypt)
    cc = CryptContext(['md5_crypt', 'bsdi_crypt', 'des_crypt'])
    c, b, a = cc.policy.iter_handlers()
    self.assertIs(a, hash.des_crypt)
    self.assertIs(b, hash.bsdi_crypt)
    self.assertIs(c, hash.md5_crypt)
    policy = cc.policy
    cc = CryptContext(policy=policy)
    self.assertEqual(cc.to_dict(), policy.to_dict())
    cc = CryptContext(policy=policy, default='bsdi_crypt')
    self.assertNotEqual(cc.to_dict(), policy.to_dict())
    self.assertEqual(cc.to_dict(), dict(schemes=['md5_crypt', 'bsdi_crypt', 'des_crypt'], default='bsdi_crypt'))
    self.assertRaises(TypeError, setattr, cc, 'policy', None)
    self.assertRaises(TypeError, CryptContext, policy='x')