from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
def test_33_options(self):
    """test internal _get_record_options() method"""

    def options(ctx, scheme, category=None):
        return ctx._config._get_record_options_with_flag(scheme, category)[0]
    cc4 = CryptContext(truncate_error=True, schemes=['sha512_crypt', 'des_crypt', 'bsdi_crypt'], deprecated=['sha512_crypt', 'des_crypt'], all__vary_rounds=0.1, bsdi_crypt__vary_rounds=0.2, sha512_crypt__max_rounds=20000, admin__context__deprecated=['des_crypt', 'bsdi_crypt'], admin__all__vary_rounds=0.05, admin__bsdi_crypt__vary_rounds=0.3, admin__sha512_crypt__max_rounds=40000)
    self.assertEqual(cc4._config.categories, ('admin',))
    self.assertEqual(options(cc4, 'sha512_crypt'), dict(deprecated=True, vary_rounds=0.1, max_rounds=20000))
    self.assertEqual(options(cc4, 'sha512_crypt', 'user'), dict(deprecated=True, vary_rounds=0.1, max_rounds=20000))
    self.assertEqual(options(cc4, 'sha512_crypt', 'admin'), dict(vary_rounds=0.05, max_rounds=40000))
    self.assertEqual(options(cc4, 'des_crypt'), dict(deprecated=True, truncate_error=True))
    self.assertEqual(options(cc4, 'des_crypt', 'user'), dict(deprecated=True, truncate_error=True))
    self.assertEqual(options(cc4, 'des_crypt', 'admin'), dict(deprecated=True, truncate_error=True))
    self.assertEqual(options(cc4, 'bsdi_crypt'), dict(vary_rounds=0.2))
    self.assertEqual(options(cc4, 'bsdi_crypt', 'user'), dict(vary_rounds=0.2))
    self.assertEqual(options(cc4, 'bsdi_crypt', 'admin'), dict(vary_rounds=0.3, deprecated=True))