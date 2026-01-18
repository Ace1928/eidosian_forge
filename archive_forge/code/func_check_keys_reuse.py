from __future__ import absolute_import
import decimal
from unittest import TestCase
import sys
import simplejson as json
from simplejson.compat import StringIO, b, binary_type
from simplejson import OrderedDict
def check_keys_reuse(self, source, loads):
    rval = loads(source)
    (a, b), (c, d) = (sorted(rval[0]), sorted(rval[1]))
    self.assertIs(a, c)
    self.assertIs(b, d)