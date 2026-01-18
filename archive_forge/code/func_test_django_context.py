from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_django_context(self):
    ctx = apps.django_context
    for hash in ['pbkdf2_sha256$29000$ZsgquwnCyBs2$fBxRQpfKd2PIeMxtkKPy0h7SrnrN+EU/cm67aitoZ2s=']:
        self.assertTrue(ctx.verify('test', hash))
    self.assertEqual(ctx.identify('!'), 'django_disabled')
    self.assertFalse(ctx.verify('test', '!'))