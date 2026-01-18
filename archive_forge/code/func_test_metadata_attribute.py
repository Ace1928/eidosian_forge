from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_metadata_attribute(self):
    res = self.sot
    self.assertTrue(hasattr(res, 'metadata'))