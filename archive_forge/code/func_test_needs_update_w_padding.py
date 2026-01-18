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
def test_needs_update_w_padding(self):
    """needs_update corrects bcrypt padding"""
    bcrypt = self.handler.using(rounds=4)
    BAD1 = '$2a$04$yjDgE74RJkeqC0/1NheSScrvKeu9IbKDpcQf/Ox3qsrRS/Kw42qIS'
    GOOD1 = '$2a$04$yjDgE74RJkeqC0/1NheSSOrvKeu9IbKDpcQf/Ox3qsrRS/Kw42qIS'
    self.assertTrue(bcrypt.needs_update(BAD1))
    self.assertFalse(bcrypt.needs_update(GOOD1))