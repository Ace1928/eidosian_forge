import unittest
from cvxpy.utilities.versioning import Version
def test_local_version_identifiers(self):
    self.assertTrue(Version('1.0.0') == Version('1.0.0+1'))
    self.assertTrue(Version('1.0.0') == Version('1.0.0+xxx'))
    self.assertTrue(Version('1.0.0') == Version('1.0.0+x.y.z'))