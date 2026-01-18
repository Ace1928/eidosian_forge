import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_needs_update_w_type(self):
    handler = self.handler
    hash = handler.hash('stub')
    self.assertFalse(handler.needs_update(hash))
    hash2 = re.sub('\\$argon2\\w+\\$', '$argon2d$', hash)
    self.assertTrue(handler.needs_update(hash2))