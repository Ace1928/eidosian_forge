from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
@skip_if_speedups_missing
def test_int_as_string_bitcount_overflow(self):
    long_count = long_type(2) ** 32 + 31

    def test():
        encoder.JSONEncoder(int_as_string_bitcount=long_count).encode(0)
    self.assertRaises((TypeError, OverflowError), test)