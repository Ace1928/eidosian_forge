from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_network_interface_no_groupset(self):
    self.mock_create_network_interface(security_groups=None)
    stack = self.create_stack(self.test_template_no_groupset)
    self.mockclient.create_port.assert_called_once_with({'port': self._port})
    stack.delete()
    self.mockclient.delete_port.assert_called_once_with('dddd')
    self.mockclient.show_subnet.assert_called_once_with('cccc')