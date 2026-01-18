from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_lowercase_v6_ethertype(self):
    arglist = ['--ethertype', 'ipv6', self._security_group.id]
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertEqual('IPv6', parsed_args.ethertype)