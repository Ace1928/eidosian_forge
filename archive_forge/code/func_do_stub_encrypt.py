import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def do_stub_encrypt(self, handler=None, **settings):
    if self.backend == 'argon2_cffi':
        handler = (handler or self.handler).using(**settings)
        self = handler(use_defaults=True)
        self.checksum = self._stub_checksum
        assert self.checksum
        return self.to_string()
    else:
        return super(_base_argon2_test, self).do_stub_encrypt(handler, **settings)