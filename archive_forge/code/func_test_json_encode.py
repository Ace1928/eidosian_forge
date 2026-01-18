import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
def test_json_encode(self):
    self.assertEqual(json_decode(json_encode('é')), 'é')
    if bytes is str:
        self.assertEqual(json_decode(json_encode(utf8('é'))), 'é')
        self.assertRaises(UnicodeDecodeError, json_encode, b'\xe9')