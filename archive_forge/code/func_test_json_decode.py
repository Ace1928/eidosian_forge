import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_json_decode(self):
    self.assertEqual(json_decode(b'"foo"'), 'foo')
    self.assertEqual(json_decode('"foo"'), 'foo')
    self.assertEqual(json_decode(utf8('"é"')), 'é')