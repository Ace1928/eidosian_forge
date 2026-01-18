from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.handlers.bcrypt import IDENT_2, IDENT_2X
from passlib.utils import repeat_string, to_bytes, is_safe_crypt_input
from passlib.utils.compat import irange, PY3
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE
def test_90_bcrypt_padding(self):
    """test passlib correctly handles bcrypt padding bits"""
    self.require_TEST_MODE('full')
    bcrypt = self.handler
    corr_desc = '.*incorrectly set padding bits'

    def check_padding(hash):
        assert hash.startswith(('$2a$', '$2b$')) and len(hash) >= 28, 'unexpectedly malformed hash: %r' % (hash,)
        self.assertTrue(hash[28] in '.Oeu', 'unused bits incorrectly set in hash: %r' % (hash,))
    for i in irange(6):
        check_padding(bcrypt.genconfig())
    for i in irange(3):
        check_padding(bcrypt.using(rounds=bcrypt.min_rounds).hash('bob'))
    with self.assertWarningList(['salt too large', corr_desc]):
        hash = bcrypt.genconfig(salt='.' * 21 + 'A.', rounds=5, relaxed=True)
    self.assertEqual(hash, '$2b$05$' + '.' * (22 + 31))
    samples = self.known_incorrect_padding
    for pwd, bad, good in samples:
        with self.assertWarningList([corr_desc]):
            self.assertEqual(bcrypt.genhash(pwd, bad), good)
        with self.assertWarningList([]):
            self.assertEqual(bcrypt.genhash(pwd, good), good)
        with self.assertWarningList([corr_desc]):
            self.assertTrue(bcrypt.verify(pwd, bad))
        with self.assertWarningList([]):
            self.assertTrue(bcrypt.verify(pwd, good))
        with self.assertWarningList([corr_desc]):
            self.assertEqual(bcrypt.normhash(bad), good)
        with self.assertWarningList([]):
            self.assertEqual(bcrypt.normhash(good), good)
    self.assertEqual(bcrypt.normhash('$md5$abc'), '$md5$abc')