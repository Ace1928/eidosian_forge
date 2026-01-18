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
def test_baremetal_create_with_driver(self):
    arglist = copy.copy(self.arglist)
    verifylist = copy.copy(self.verifylist)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = copy.copy(self.collist)
    self.assertEqual(collist, columns)
    self.assertNotIn('ports', columns)
    self.assertNotIn('states', columns)
    datalist = copy.copy(self.datalist)
    self.assertEqual(datalist, tuple(data))
    kwargs = copy.copy(self.actual_kwargs)
    self.baremetal_mock.node.create.assert_called_once_with(**kwargs)