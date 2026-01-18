from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_90_special(self):
    """test marker option & special behavior"""
    warnings.filterwarnings('ignore', 'passing settings to .*.hash\\(\\) is deprecated')
    handler = self.handler
    self.assertEqual(handler.genhash('stub', '!asd'), '!asd')
    self.assertEqual(handler.genhash('stub', ''), handler.default_marker)
    self.assertEqual(handler.hash('stub'), handler.default_marker)
    self.assertEqual(handler.using().default_marker, handler.default_marker)
    self.assertEqual(handler.genhash('stub', '', marker='*xxx'), '*xxx')
    self.assertEqual(handler.hash('stub', marker='*xxx'), '*xxx')
    self.assertEqual(handler.using(marker='*xxx').hash('stub'), '*xxx')
    self.assertRaises(ValueError, handler.genhash, 'stub', '', marker='abc')
    self.assertRaises(ValueError, handler.hash, 'stub', marker='abc')
    self.assertRaises(ValueError, handler.using, marker='abc')