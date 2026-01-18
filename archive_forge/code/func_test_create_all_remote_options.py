from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_all_remote_options(self):
    arglist = ['--remote-ip', '10.10.0.0/24', '--remote-group', self._security_group.id, '--remote-address-group', self._address_group.id, self._security_group.id]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])