from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
def test_13_repr(self):
    """test repr()"""
    h = uh.PrefixWrapper('h2', 'md5_crypt', '{XXX}', orig_prefix='$1$')
    self.assertRegex(repr(h), '(?x)^PrefixWrapper\\(\n                [\'"]h2[\'"],\\s+\n                [\'"]md5_crypt[\'"],\\s+\n                prefix=u?["\']{XXX}[\'"],\\s+\n                orig_prefix=u?["\']\\$1\\$[\'"]\n            \\)$')