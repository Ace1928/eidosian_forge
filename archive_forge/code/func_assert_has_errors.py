import textwrap
from unittest import mock
import pycodestyle
import testtools
from keystoneauth1.hacking import checks
from keystoneauth1.tests.unit import keystoneauth_fixtures
def assert_has_errors(self, code, expected_errors=None):
    actual_errors = [e[:3] for e in self.run_check(code)]
    self.assertEqual(expected_errors or [], actual_errors)