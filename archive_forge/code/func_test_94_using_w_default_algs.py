import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_94_using_w_default_algs(self, param='default_algs'):
    """using() -- 'default_algs' parameter"""
    handler = self.handler
    orig = list(handler.default_algs)
    subcls = handler.using(**{param: 'sha1,md5'})
    self.assertEqual(handler.default_algs, orig)
    self.assertEqual(subcls.default_algs, ['md5', 'sha-1'])
    h1 = subcls.hash('dummy')
    self.assertEqual(handler.extract_digest_algs(h1), ['md5', 'sha-1'])