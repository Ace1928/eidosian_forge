from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_bad_bool_args(self):

    def test(name):
        encoder.JSONEncoder(**{name: BadBool()}).encode({})
    self.assertRaises(ZeroDivisionError, test, 'skipkeys')
    self.assertRaises(ZeroDivisionError, test, 'ensure_ascii')
    self.assertRaises(ZeroDivisionError, test, 'check_circular')
    self.assertRaises(ZeroDivisionError, test, 'allow_nan')
    self.assertRaises(ZeroDivisionError, test, 'sort_keys')
    self.assertRaises(ZeroDivisionError, test, 'use_decimal')
    self.assertRaises(ZeroDivisionError, test, 'namedtuple_as_object')
    self.assertRaises(ZeroDivisionError, test, 'tuple_as_array')
    self.assertRaises(ZeroDivisionError, test, 'bigint_as_string')
    self.assertRaises(ZeroDivisionError, test, 'for_json')
    self.assertRaises(ZeroDivisionError, test, 'ignore_nan')
    self.assertRaises(ZeroDivisionError, test, 'iterable_as_array')