import os.path
import testtools
from fixtures import PythonPackage, TestWithFixtures
def test_has_tempdir(self):
    fixture = PythonPackage('foo', [])
    fixture.setUp()
    try:
        self.assertTrue(os.path.isdir(fixture.base))
    finally:
        fixture.cleanUp()