from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_get_entity_get_and_search(self):
    resources = ['flavor', 'image', 'volume', 'network', 'subnet', 'port', 'floating_ip', 'security_group']
    for r in resources:
        self.assertTrue(hasattr(self.cloud, 'get_%s_by_id' % r))
        self.assertTrue(hasattr(self.cloud, 'search_%ss' % r))