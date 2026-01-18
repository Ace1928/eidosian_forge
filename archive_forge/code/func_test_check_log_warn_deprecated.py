import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_log_warn_deprecated(self):
    bad = "LOG.warn('i am deprecated!')"
    good = "LOG.warning('zlatan is the best')"
    f = tc.check_log_warn_deprecated
    self.assertLineFails(f, bad, '')
    self.assertLinePasses(f, good, '')