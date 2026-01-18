import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
def test_create_unknown_proxy(self):
    self.register_uris([self.get_placement_discovery_mock_dict()])

    def closure():
        return self.cloud.placement
    self.assertThat(closure, matchers.Warnings(matchers.HasLength(0)))
    self.assertIsInstance(self.cloud.placement, proxy.Proxy)
    self.assert_calls()