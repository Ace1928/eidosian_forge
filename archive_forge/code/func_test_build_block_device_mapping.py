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
def test_build_block_device_mapping(self):
    self.assertIsNone(servers.Server._build_block_device_mapping([]))
    self.assertIsNone(servers.Server._build_block_device_mapping(None))
    self.assertEqual({'vda': '1234::', 'vdb': '1234:snap:'}, servers.Server._build_block_device_mapping([{'device_name': 'vda', 'volume_id': '1234'}, {'device_name': 'vdb', 'snapshot_id': '1234'}]))
    self.assertEqual({'vdc': '1234::10', 'vdd': '1234:snap::True'}, servers.Server._build_block_device_mapping([{'device_name': 'vdc', 'volume_id': '1234', 'volume_size': 10}, {'device_name': 'vdd', 'snapshot_id': '1234', 'delete_on_termination': True}]))