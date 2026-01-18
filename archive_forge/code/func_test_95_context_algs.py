import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_95_context_algs(self):
    """test handling of 'algs' in context object"""
    handler = self.handler
    from passlib.context import CryptContext
    c1 = CryptContext(['scram'], scram__algs='sha1,md5')
    h = c1.hash('dummy')
    self.assertEqual(handler.extract_digest_algs(h), ['md5', 'sha-1'])
    self.assertFalse(c1.needs_update(h))
    c2 = c1.copy(scram__algs='sha1')
    self.assertFalse(c2.needs_update(h))
    c2 = c1.copy(scram__algs='sha1,sha256')
    self.assertTrue(c2.needs_update(h))