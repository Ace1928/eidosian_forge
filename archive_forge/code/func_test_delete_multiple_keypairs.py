import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_keypairs(self):
    arglist = []
    for k in self.keypairs:
        arglist.append(k.name)
    verifylist = [('name', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for k in self.keypairs:
        calls.append(call(k.name, ignore_missing=False))
    self.compute_sdk_client.delete_keypair.assert_has_calls(calls)
    self.assertIsNone(result)