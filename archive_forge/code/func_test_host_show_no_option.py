from unittest import mock
from openstackclient.compute.v2 import host
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils as tests_utils
def test_host_show_no_option(self, h_mock):
    h_mock.host_show.return_value = [self._host]
    arglist = []
    verifylist = []
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)