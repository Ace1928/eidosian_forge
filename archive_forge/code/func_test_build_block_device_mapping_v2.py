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
def test_build_block_device_mapping_v2(self):
    self.assertIsNone(servers.Server._build_block_device_mapping_v2([]))
    self.assertIsNone(servers.Server._build_block_device_mapping_v2(None))
    self.assertEqual([{'uuid': '1', 'source_type': 'volume', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}], servers.Server._build_block_device_mapping_v2([{'volume_id': '1'}]))
    self.assertEqual([{'uuid': '1', 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}], servers.Server._build_block_device_mapping_v2([{'snapshot_id': '1'}]))
    self.assertEqual([{'uuid': '1', 'source_type': 'image', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}], servers.Server._build_block_device_mapping_v2([{'image': '1'}]))
    self.assertEqual([{'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True, 'guest_format': 'swap', 'volume_size': 1}], servers.Server._build_block_device_mapping_v2([{'swap_size': 1}]))
    self.assertEqual([], servers.Server._build_block_device_mapping_v2([{'device_name': ''}]))
    self.assertEqual([{'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'delete_on_termination': True, 'volume_size': 1, 'guest_format': 'ext4'}], servers.Server._build_block_device_mapping_v2([{'ephemeral_size': 1, 'ephemeral_format': 'ext4'}]))