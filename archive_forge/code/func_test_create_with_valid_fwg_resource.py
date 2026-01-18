import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_create_with_valid_fwg_resource(self):
    name = self.res['name']
    resource_id = 'valid_fwg_id'
    resource_type = RES_TYPE_FWG
    with mock.patch.object(self.neutronclient, 'find_resource', return_value={'id': resource_id}):
        arglist = [name, '--resource-type', resource_type, '--resource', resource_id]
        verifylist = [('name', name), ('resource_type', resource_type), ('resource', resource_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        expect = {'name': self.res['name'], 'resource_type': RES_TYPE_FWG, 'resource_id': 'valid_fwg_id'}
        self.neutronclient.find_resource.assert_called_with(resource_type, resource_id, cmd_resource='fwaas_firewall_group')
        self.mocked.assert_called_once_with({'log': expect})
        self.assertEqual(self.ordered_headers, headers)
        self.assertEqual(self.ordered_data, data)