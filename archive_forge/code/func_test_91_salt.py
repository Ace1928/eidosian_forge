from __future__ import absolute_import, division, print_function
import logging
from passlib import hash, exc
from passlib.utils.compat import u
from .utils import UserHandlerMixin, HandlerCase, repeat_string
from .test_handlers import UPASS_TABLE
def test_91_salt(self):
    """test salt value border cases"""
    handler = self.handler
    self.assertRaises(TypeError, handler, salt=None)
    handler(salt=None, use_defaults=True)
    self.assertRaises(TypeError, handler, salt='abc')
    self.assertRaises(ValueError, handler, salt=-10)
    self.assertRaises(ValueError, handler, salt=100)
    self.assertRaises(TypeError, handler.using, salt='abc')
    self.assertRaises(ValueError, handler.using, salt=-10)
    self.assertRaises(ValueError, handler.using, salt=100)
    with self.assertWarningList('salt/offset must be.*'):
        subcls = handler.using(salt=100, relaxed=True)
    self.assertEqual(subcls(use_defaults=True).salt, 52)