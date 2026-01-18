import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_build_userdata_with_instance_user(self):
    """Add a custom instance user."""
    cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
    data = self.nova_plugin.build_userdata({}, instance_user='ec2-user')
    self.assertIn('user: ', data)
    self.assertIn('useradd', data)
    self.assertIn('ec2-user', data)