import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.allocation.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
    self.assertEqual(datalist, tuple(data))