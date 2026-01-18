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
def test_build_nics_without_internal_port(self):
    tmpl = '\n        heat_template_version: 2015-10-15\n        resources:\n          server:\n            type: OS::Nova::Server\n            properties:\n              flavor: m1.small\n              image: F17-x86_64-gold\n              networks:\n                - port: 12345\n                  network: 4321\n        '
    t, stack, server = self._return_template_stack_and_rsrc_defn('test', tmpl)
    create_internal_port = self.patchobject(server, '_create_internal_port', return_value='12345')
    networks = [{'port': '12345', 'network': '4321'}]
    nics = server._build_nics(networks)
    self.assertEqual([{'port-id': '12345', 'net-id': '4321'}], nics)
    self.assertEqual(0, create_internal_port.call_count)