from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_ldap_nocrypt_context(self):
    ctx = apps.ldap_nocrypt_context
    for hash in ['{SSHA}cPusOzd6d5n3OjSVK3R329ZGCNyFcC7F', 'test']:
        self.assertTrue(ctx.verify('test', hash))
    self.assertIs(ctx.identify('{CRYPT}$5$rounds=31817$iZGmlyBQ99JSB5n6$p4E.pdPBWx19OajgjLRiOW0itGnyxDGgMlDcOsfaI17'), None)