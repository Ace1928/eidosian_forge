import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_baremetal_list_bios_setting_fields(self):
    arglist = ['node_uuid', '--fields', 'name', 'attribute_type']
    verifylist = [('fields', [['name', 'attribute_type']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.baremetal_mock.node.list_bios_settings.return_value = baremetal_fakes.BIOS_DETAILED_SETTINGS
    columns, data = self.cmd.take_action(parsed_args)
    self.assertNotIn('Value', columns)
    self.assertIn('Name', columns)
    self.assertIn('Attribute Type', columns)
    kwargs = {'detail': False, 'fields': ('name', 'attribute_type')}
    self.baremetal_mock.node.list_bios_settings.assert_called_with('node_uuid', **kwargs)