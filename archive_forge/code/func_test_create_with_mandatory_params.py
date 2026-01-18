import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_create_with_mandatory_params(self):
    name = self.res['name']
    arglist = [name, '--resource-type', RES_TYPE_SG]
    verifylist = [('name', name), ('resource_type', RES_TYPE_SG)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    expect = {'name': self.res['name'], 'resource_type': self.res['resource_type']}
    self.mocked.assert_called_once_with({'log': expect})
    self.assertEqual(self.ordered_headers, headers)
    self.assertEqual(self.ordered_data, data)