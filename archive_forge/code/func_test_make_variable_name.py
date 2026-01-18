from oslotest import base as test_base
import testscenarios.testcase
from oslo_i18n import _locale
def test_make_variable_name(self):
    var = _locale.get_locale_dir_variable_name(self.domain)
    self.assertEqual(self.expected, var)