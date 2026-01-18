from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_asdict_unbound_method_dumps(self):
    for f in CONSTRUCTORS:
        self.assertEqual(json.dumps(f(Value), default=lambda v: v.__name__), json.dumps(f(Value.__name__)))