from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_namedtuple_dumps_false(self):
    for v in [Value(1), Point(1, 2)]:
        l = list(v)
        self.assertEqual(l, json.loads(json.dumps(v, namedtuple_as_object=False)))
        self.assertRaises(TypeError, json.dumps, v, tuple_as_array=False, namedtuple_as_object=False)