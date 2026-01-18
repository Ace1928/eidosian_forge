from unittest import TestCase
import simplejson as json
def test_defaultrecursion(self):
    enc = RecursiveJSONEncoder()
    self.assertEqual(enc.encode(JSONTestObject), '"JSONTestObject"')
    enc.recurse = True
    try:
        enc.encode(JSONTestObject)
    except ValueError:
        pass
    else:
        self.fail("didn't raise ValueError on default recursion")