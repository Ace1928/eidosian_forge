import uuid
from designateclient.tests import base
def assertList(self, expected, actual):
    self.assertEqual(len(expected), len(actual))
    for i in expected:
        self.assertIn(i, actual)