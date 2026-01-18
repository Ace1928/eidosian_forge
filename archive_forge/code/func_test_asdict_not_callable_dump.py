from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_asdict_not_callable_dump(self):
    for f in CONSTRUCTORS:
        self.assertRaises(TypeError, json.dump, f(DeadDuck()), StringIO(), namedtuple_as_object=True)
        sio = StringIO()
        json.dump(f(DeadDict()), sio, namedtuple_as_object=True)
        self.assertEqual(json.dumps(f({})), sio.getvalue())
        self.assertRaises(TypeError, json.dump, f(Value), StringIO(), namedtuple_as_object=True)