from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import hosts, hash as hashmod
from passlib.utils import unix_crypt_schemes
from passlib.tests.utils import TestCase
def check_unix_disabled(self, ctx):
    for hash in ['', '!', '*', '!$1$TXl/FX/U$BZge.lr.ux6ekjEjxmzwz0']:
        self.assertEqual(ctx.identify(hash), 'unix_disabled')
        self.assertFalse(ctx.verify('test', hash))