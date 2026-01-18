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
def test_baremetal_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.node.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Name', 'Instance UUID', 'Power State', 'Provisioning State', 'Maintenance')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_instance_uuid, baremetal_fakes.baremetal_power_state, baremetal_fakes.baremetal_provision_state, baremetal_fakes.baremetal_maintenance),)
    self.assertEqual(datalist, tuple(data))