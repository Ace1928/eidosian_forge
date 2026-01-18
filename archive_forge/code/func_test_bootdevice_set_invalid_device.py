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
def test_bootdevice_set_invalid_device(self):
    arglist = ['node_uuid', 'foo']
    verifylist = [('nodes', ['node_uuid']), ('device', 'foo')]
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)