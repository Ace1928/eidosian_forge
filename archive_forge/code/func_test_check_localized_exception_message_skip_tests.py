import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_localized_exception_message_skip_tests(self):
    f = tc.check_raised_localized_exceptions
    self.assertLinePasses(f, "raise KeyError('Error text')", 'neutron_lib/tests/unit/mytest.py')