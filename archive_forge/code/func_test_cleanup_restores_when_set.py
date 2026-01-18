import os
import testtools
from fixtures import EnvironmentVariable, TestWithFixtures
def test_cleanup_restores_when_set(self):
    fixture = EnvironmentVariable('FIXTURES_TEST_VAR')
    os.environ['FIXTURES_TEST_VAR'] = 'bar'
    with fixture:
        os.environ['FIXTURES_TEST_VAR'] = 'quux'
    self.assertEqual('bar', os.environ.get('FIXTURES_TEST_VAR'))