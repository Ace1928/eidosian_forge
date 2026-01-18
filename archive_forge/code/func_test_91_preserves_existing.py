from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
def test_91_preserves_existing(self):
    """test preserves existing disabled hash"""
    handler = self.handler
    self.assertEqual(handler.genhash('stub', ''), '!')
    self.assertEqual(handler.hash('stub'), '!')
    self.assertEqual(handler.genhash('stub', '!asd'), '!asd')