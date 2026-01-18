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
def test_internet_gateway(self):
    stack = self.create_stack(self.test_template)
    self.mockclient.show_network.assert_called_with('aaaa')
    self.mockclient.create_network.assert_called_with({'network': {'name': self.vpc_name}})
    self.assertEqual(2, self.mockclient.create_router.call_count)
    self.mockclient.create_router.assert_called_with({'router': {'name': self.rt_name}})
    self.mockclient.add_interface_router.assert_has_calls([mock.call('bbbb', {'subnet_id': u'cccc'}), mock.call('ffff', {'subnet_id': u'cccc'})])
    self.mockclient.list_networks.assert_called_once_with(**{'router:external': True})
    gateway = stack['the_gateway']
    self.assertResourceState(gateway, gateway.physical_resource_name())
    self.mockclient.add_gateway_router.assert_called_with('ffff', {'network_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'})
    attachment = stack['the_attachment']
    self.assertResourceState(attachment, 'the_attachment')
    route_table = stack['the_route_table']
    self.assertEqual([route_table], list(attachment._vpc_route_tables()))
    stack.delete()
    self.mockclient.remove_interface_router.assert_has_calls([mock.call('ffff', {'subnet_id': u'cccc'}), mock.call('bbbb', {'subnet_id': u'cccc'})])
    self.mockclient.remove_gateway_router.assert_called_with('ffff')
    self.assertEqual(2, self.mockclient.remove_gateway_router.call_count)
    self.assertEqual(2, self.mockclient.show_subnet.call_count)
    self.mockclient.show_subnet.assert_called_with('cccc')
    self.mockclient.show_router.assert_called_with('ffff')
    self.assertEqual(2, self.mockclient.show_router.call_count)
    self.mockclient.delete_router.assert_called_once_with('ffff')