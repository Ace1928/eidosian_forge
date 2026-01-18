import os.path
import testtools
from fixtures import PythonPackage, TestWithFixtures
def test_no__init__(self):
    fixture = PythonPackage('foo', [('bar.py', b'woo')], init=False)
    fixture.setUp()
    try:
        self.assertFalse(os.path.exists(os.path.join(fixture.base, 'foo', '__init__.py')))
    finally:
        fixture.cleanUp()