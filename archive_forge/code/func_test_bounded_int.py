from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def test_bounded_int(self):
    max_str_digits = getattr(sys, 'get_int_max_str_digits', lambda: 4300)()
    s = '1' + '0' * (max_str_digits - 1)
    self.assertEqual(json.loads(s), int(s))
    self.assertRaises(ValueError, json.loads, s + '0')