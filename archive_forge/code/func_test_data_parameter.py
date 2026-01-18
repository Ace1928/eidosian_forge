import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
def test_data_parameter(self):
    handler = self.handler
    sample1 = '$argon2i$v=19$m=512,t=2,p=2,data=c29tZWRhdGE$c29tZXNhbHQ$KgHyCesFyyjkVkihZ5VNFw'
    sample2 = '$argon2i$v=19$m=512,t=2,p=2,data=c29tZWRhdGE$c29tZXNhbHQ$uEeXt1dxN1iFKGhklseW4w'
    sample3 = '$argon2i$v=19$m=512,t=2,p=2$c29tZXNhbHQ$uEeXt1dxN1iFKGhklseW4w'
    if self.backend == 'argon2_cffi':
        self.assertRaises(NotImplementedError, handler.verify, 'password', sample1)
        self.assertEqual(handler.genhash('password', sample1), sample3)
    else:
        assert self.backend == 'argon2pure'
        self.assertTrue(handler.verify('password', sample1))
        self.assertEqual(handler.genhash('password', sample1), sample1)
    if self.backend == 'argon2_cffi':
        self.assertRaises(NotImplementedError, handler.verify, 'password', sample2)
        self.assertEqual(handler.genhash('password', sample1), sample3)
    else:
        assert self.backend == 'argon2pure'
        self.assertFalse(self.handler.verify('password', sample2))
        self.assertEqual(handler.genhash('password', sample2), sample1)