from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import hosts, hash as hashmod
from passlib.utils import unix_crypt_schemes
from passlib.tests.utils import TestCase
def test_linux_context(self):
    ctx = hosts.linux_context
    for hash in ['$6$rounds=41128$VoQLvDjkaZ6L6BIE$4pt.1Ll1XdDYduEwEYPCMOBiR6W6znsyUEoNlcVXpv2gKKIbQolgmTGe6uEEVJ7azUxuc8Tf7zV9SD2z7Ij751', '$5$rounds=31817$iZGmlyBQ99JSB5n6$p4E.pdPBWx19OajgjLRiOW0itGnyxDGgMlDcOsfaI17', '$1$TXl/FX/U$BZge.lr.ux6ekjEjxmzwz0', 'kAJJz.Rwp0A/I']:
        self.assertTrue(ctx.verify('test', hash))
    self.check_unix_disabled(ctx)