from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
def skip_if_speedups_missing(func):

    def wrapper(*args, **kwargs):
        if not has_speedups():
            if hasattr(unittest, 'SkipTest'):
                raise unittest.SkipTest('C Extension not available')
            else:
                sys.stdout.write('C Extension not available')
                return
        return func(*args, **kwargs)
    return wrapper