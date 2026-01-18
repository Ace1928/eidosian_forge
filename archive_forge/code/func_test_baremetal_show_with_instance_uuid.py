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
def test_baremetal_show_with_instance_uuid(self):
    arglist = ['xxx-xxxxxx-xxxx', '--instance']
    verifylist = [('instance_uuid', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = ['xxx-xxxxxx-xxxx']
    self.baremetal_mock.node.get_by_instance_uuid.assert_called_with(*args, fields=None)