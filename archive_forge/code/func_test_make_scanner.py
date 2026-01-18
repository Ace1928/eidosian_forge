from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_make_scanner(self):
    self.assertRaises(AttributeError, scanner.c_make_scanner, 1)