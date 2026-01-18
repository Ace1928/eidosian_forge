import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_cleanup_deletes_when_missing(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
    os.environ.pop('FIXTURES_TEST_VAR', '')
    with fixture:
        os.environ['FIXTURES_TEST_VAR'] = 'foo'
    self.assertEqual(None, os.environ.get('FIXTURES_TEST_VAR'))