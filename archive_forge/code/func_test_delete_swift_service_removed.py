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
def test_delete_swift_service_removed(self):
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    return_server = self.fc.servers.list()[1]
    stack_name = 'software_config_s'
    tmpl, stack = self._setup_test_stack(stack_name)
    props = tmpl.t['Resources']['WebServer']['Properties']
    props['user_data_format'] = 'SOFTWARE_CONFIG'
    props['software_config_transport'] = 'POLL_TEMP_URL'
    self.server_props = props
    resource_defns = tmpl.resource_definitions(stack)
    server = servers.Server('WebServer', resource_defns['WebServer'], stack)
    self.patchobject(server, 'store_external_ports')
    sc = mock.Mock()
    sc.head_account.return_value = {'x-account-meta-temp-url-key': 'secrit'}
    sc.url = 'http://192.0.2.2'
    self.patchobject(swift.SwiftClientPlugin, '_create', return_value=sc)
    self.patchobject(self.fc.servers, 'create', return_value=return_server)
    scheduler.TaskRunner(server.create)()
    self.assertEqual((server.CREATE, server.COMPLETE), server.state)
    self.patchobject(server.client_plugin(), 'does_endpoint_exist', return_value=False)
    side_effect = [server, fakes_nova.fake_exception()]
    self.patchobject(self.fc.servers, 'get', side_effect=side_effect)
    scheduler.TaskRunner(server.delete)()
    self.assertEqual((server.DELETE, server.COMPLETE), server.state)