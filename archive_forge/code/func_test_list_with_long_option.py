import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_list_with_long_option(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with()
    self.assertEqual(list(self.headers), headers)