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
def test_network_interface_existing_groupset(self):
    self.patchobject(parser.Stack, 'resource_by_refid')
    self.mock_create_network_interface()
    stack = self.create_stack(self.test_template)
    self.validate_mock_create_network()
    self.mockclient.create_port.assert_called_once_with({'port': self._port})
    self.mockclient.create_security_group.assert_called_once_with({'security_group': {'name': self.sg_name, 'description': 'SSH access'}})
    self.mockclient.create_security_group_rule.assert_called_once_with(self.create_security_group_rule_expected)
    try:
        self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
        rsrc = stack['the_nic']
        self.assertResourceState(rsrc, 'dddd')
    finally:
        stack.delete()
    self.mockclient.delete_port.assert_called_once_with('dddd')
    self.mockclient.show_subnet.assert_called_once_with('cccc')
    self.mockclient.show_security_group.assert_called_once_with(self._group)
    self.mockclient.delete_security_group.assert_called_once_with('0389f747-7785-4757-b7bb-2ab07e4b09c3')
    self.mockclient.delete_security_group_rule.assert_called_once_with('bbbb')