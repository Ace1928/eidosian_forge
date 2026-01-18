from __future__ import with_statement
from logging import getLogger
import warnings
import sys
from passlib import hash, registry, exc
from passlib.registry import register_crypt_handler, register_crypt_handler_path, \
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase
def test_get_crypt_handler(self):
    """test get_crypt_handler()"""

    class dummy_1(uh.StaticHandler):
        name = 'dummy_1'
    self.assertRaises(KeyError, get_crypt_handler, 'dummy_1')
    self.assertIs(get_crypt_handler('dummy_1', None), None)
    register_crypt_handler(dummy_1)
    self.assertIs(get_crypt_handler('dummy_1'), dummy_1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'handler names should be lower-case, and use underscores instead of hyphens:.*', UserWarning)
        self.assertIs(get_crypt_handler('DUMMY-1'), dummy_1)
        register_crypt_handler_path('dummy_0', __name__)
        self.assertIs(get_crypt_handler('DUMMY-0'), dummy_0)
    from passlib import hash
    hash.__dict__['_fake'] = 'dummy'
    for name in ['_fake', '__package__']:
        self.assertRaises(KeyError, get_crypt_handler, name)
        self.assertIs(get_crypt_handler(name, None), None)