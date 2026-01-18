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
def test_get_network_id_neutron(self):
    return_server = self.fc.servers.list()[3]
    server = self._create_test_server(return_server, 'networks_update')
    net = {'port': '2a60cbaa-3d33-4af6-a9ce-83594ac546fc'}
    net_id = server._get_network_id(net)
    self.assertIsNone(net_id)
    net = {'network': 'f3ef5d2f-d7ba-4b27-af66-58ca0b81e032', 'fixed_ip': '1.2.3.4'}
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='f3ef5d2f-d7ba-4b27-af66-58ca0b81e032')
    net_id = server._get_network_id(net)
    self.assertEqual('f3ef5d2f-d7ba-4b27-af66-58ca0b81e032', net_id)
    net = {'network': '', 'fixed_ip': '1.2.3.4'}
    net_id = server._get_network_id(net)
    self.assertIsNone(net_id)