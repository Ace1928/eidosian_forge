from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_custom_app_context(self):
    ctx = apps.custom_app_context
    self.assertEqual(ctx.schemes(), ('sha512_crypt', 'sha256_crypt'))
    for hash in ['$6$rounds=41128$VoQLvDjkaZ6L6BIE$4pt.1Ll1XdDYduEwEYPCMOBiR6W6znsyUEoNlcVXpv2gKKIbQolgmTGe6uEEVJ7azUxuc8Tf7zV9SD2z7Ij751', '$5$rounds=31817$iZGmlyBQ99JSB5n6$p4E.pdPBWx19OajgjLRiOW0itGnyxDGgMlDcOsfaI17']:
        self.assertTrue(ctx.verify('test', hash))