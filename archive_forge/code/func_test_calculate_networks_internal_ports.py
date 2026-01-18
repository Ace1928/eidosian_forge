import collections
import contextlib
import copy
from unittest import mock
from keystoneauth1 import exceptions as ks_exceptions
from neutronclient.v2_0 import client as neutronclient
from novaclient import exceptions as nova_exceptions
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import requests
from urllib import parse as urlparse
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.nova import server as servers
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_calculate_networks_internal_ports(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - network: 4321\n                  subnet: 1234\n                  fixed_ip: 127.0.0.1\n                - port: 3344\n        '
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
    data_mock = self.patchobject(server, '_data_get_ports')
    data_mock.side_effect = [[{'id': '1122'}], [{'id': '1122'}], []]
    self.port_create.return_value = {'port': {'id': '7788'}}
    data_set = self.patchobject(resource.Resource, 'data_set')
    old_net = [self.create_old_net(net='4321', subnet='1234', ip='127.0.0.1'), self.create_old_net(port='3344')]
    new_net = [{'port': '3344'}, {'port': '5566'}, {'network': '4321', 'subnet': '5678', 'fixed_ip': '10.0.0.1'}]
    interfaces = [create_fake_iface(port='1122', net='4321', ip='127.0.0.1', subnet='1234'), create_fake_iface(port='3344', net='4321', ip='10.0.0.2', subnet='subnet')]
    server.calculate_networks(old_net, new_net, interfaces)
    self.port_delete.assert_called_once_with('1122')
    self.port_create.assert_called_once_with({'port': {'name': 'server-port-1', 'network_id': '4321', 'fixed_ips': [{'subnet_id': '5678', 'ip_address': '10.0.0.1'}]}})
    self.assertEqual(2, data_set.call_count)
    data_set.assert_has_calls((mock.call('internal_ports', '[]'), mock.call('internal_ports', '[{"id": "7788"}]')))