import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_check_neutron_namespace_imports(self):
    f = checks.check_neutron_namespace_imports
    self.assertLinePasses(f, 'from neutron_lib import constants')
    self.assertLinePasses(f, 'import neutron_lib.constants')
    self.assertLineFails(f, 'from neutron.common import rpc')
    self.assertLineFails(f, 'from neutron import context')
    self.assertLineFails(f, 'import neutron.common.config')