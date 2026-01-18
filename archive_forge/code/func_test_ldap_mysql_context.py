from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_ldap_mysql_context(self):
    ctx = apps.mysql_context
    for hash in ['*94BDCEBE19083CE2A1F959FD02F964C7AF4CFC29', '378b243e220ca493']:
        self.assertTrue(ctx.verify('test', hash))