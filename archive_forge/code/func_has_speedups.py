from __future__ import with_statement
import sys
import unittest
from unittest import TestCase
import simplejson
from simplejson import encoder, decoder, scanner
from simplejson.compat import PY3, long_type, b
def has_speedups():
    return encoder.c_make_encoder is not None