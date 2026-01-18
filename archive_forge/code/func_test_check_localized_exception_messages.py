import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_localized_exception_messages(self):
    f = tc.check_raised_localized_exceptions
    self.assertLineFails(f, "     raise KeyError('Error text')", '')
    self.assertLineFails(f, ' raise KeyError("Error text")', '')
    self.assertLinePasses(f, ' raise KeyError(_("Error text"))', '')
    self.assertLinePasses(f, ' raise KeyError(_ERR("Error text"))', '')
    self.assertLinePasses(f, ' raise KeyError(translated_msg)', '')
    self.assertLinePasses(f, '# raise KeyError("Not translated")', '')
    self.assertLinePasses(f, 'print("raise KeyError("Not translated")")', '')