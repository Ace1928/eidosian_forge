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
def test_handle_snapshot_delete(self):
    t = template_format.parse(wp_template)
    t['Resources']['WebServer']['DeletionPolicy'] = 'Snapshot'
    tmpl = template.Template(t)
    stack = parser.Stack(utils.dummy_context(), 'snapshot_policy', tmpl)
    stack.store()
    rsrc = stack['WebServer']
    mock_plugin = self.patchobject(nova.NovaClientPlugin, 'client')
    mock_plugin.return_value = self.fc
    delete_server = self.patchobject(self.fc.servers, 'delete')
    delete_server.side_effect = nova_exceptions.NotFound(404)
    create_image = self.patchobject(self.fc.servers, 'create_image')
    self.patchobject(servers.Server, 'user_data_software_config', return_value=True)
    delete_internal_ports = self.patchobject(servers.Server, '_delete_internal_ports')
    delete_queue = self.patchobject(servers.Server, '_delete_queue')
    delete_user = self.patchobject(servers.Server, '_delete_user')
    delete_swift_object = self.patchobject(servers.Server, '_delete_temp_url')
    rsrc.handle_snapshot_delete((rsrc.CREATE, rsrc.FAILED))
    delete_server.assert_not_called()
    create_image.assert_not_called()
    delete_internal_ports.assert_called_once_with()
    delete_queue.assert_called_once_with()
    delete_user.assert_called_once_with()
    delete_swift_object.assert_called_once_with()
    rsrc.resource_id = '4567'
    rsrc.handle_snapshot_delete((rsrc.CREATE, rsrc.FAILED))
    delete_server.assert_called_once_with('4567')
    create_image.assert_not_called()
    self.assertEqual(2, delete_internal_ports.call_count)