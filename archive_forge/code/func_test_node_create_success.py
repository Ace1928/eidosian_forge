import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import node as sn
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_node_create_success(self):
    self._create_node()
    expect_kwargs = {'name': 'SenlinNode', 'profile_id': 'fake_profile_id', 'metadata': {'foo': 'bar'}, 'cluster_id': 'fake_cluster_id'}
    self.senlin_mock.create_node.assert_called_once_with(**expect_kwargs)