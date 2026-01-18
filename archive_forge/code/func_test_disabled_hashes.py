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
def test_disabled_hashes(self):
    """disabled hash support"""
    from passlib.exc import UnknownHashError
    from passlib.hash import md5_crypt, unix_disabled
    ctx = CryptContext(['des_crypt'])
    ctx2 = CryptContext(['des_crypt', 'unix_disabled'])
    h_ref = ctx.hash('foo')
    h_other = md5_crypt.hash('foo')
    self.assertRaisesRegex(RuntimeError, 'no disabled hasher present', ctx.disable)
    self.assertRaisesRegex(RuntimeError, 'no disabled hasher present', ctx.disable, h_ref)
    self.assertRaisesRegex(RuntimeError, 'no disabled hasher present', ctx.disable, h_other)
    h_dis = ctx2.disable()
    self.assertEqual(h_dis, unix_disabled.default_marker)
    h_dis_ref = ctx2.disable(h_ref)
    self.assertEqual(h_dis_ref, unix_disabled.default_marker + h_ref)
    h_dis_other = ctx2.disable(h_other)
    self.assertEqual(h_dis_other, unix_disabled.default_marker + h_other)
    self.assertEqual(ctx2.disable(h_dis_ref), h_dis_ref)
    self.assertTrue(ctx.is_enabled(h_ref))
    self.assertRaises(UnknownHashError, ctx.is_enabled, h_other)
    self.assertRaises(UnknownHashError, ctx.is_enabled, h_dis)
    self.assertRaises(UnknownHashError, ctx.is_enabled, h_dis_ref)
    self.assertTrue(ctx2.is_enabled(h_ref))
    self.assertRaises(UnknownHashError, ctx.is_enabled, h_other)
    self.assertFalse(ctx2.is_enabled(h_dis))
    self.assertFalse(ctx2.is_enabled(h_dis_ref))
    self.assertRaises(UnknownHashError, ctx.enable, '')
    self.assertRaises(TypeError, ctx.enable, None)
    self.assertEqual(ctx.enable(h_ref), h_ref)
    self.assertRaises(UnknownHashError, ctx.enable, h_other)
    self.assertRaises(UnknownHashError, ctx.enable, h_dis)
    self.assertRaises(UnknownHashError, ctx.enable, h_dis_ref)
    self.assertRaises(UnknownHashError, ctx.enable, '')
    self.assertRaises(TypeError, ctx2.enable, None)
    self.assertEqual(ctx2.enable(h_ref), h_ref)
    self.assertRaises(UnknownHashError, ctx2.enable, h_other)
    self.assertRaisesRegex(ValueError, 'cannot restore original hash', ctx2.enable, h_dis)
    self.assertEqual(ctx2.enable(h_dis_ref), h_ref)