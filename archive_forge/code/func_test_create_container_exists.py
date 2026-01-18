import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
def test_create_container_exists(self):
    """Test creating a container that exists."""
    self.register_uris([dict(method='HEAD', uri=self.container_endpoint, headers={'Content-Length': '0', 'X-Container-Object-Count': '0', 'Accept-Ranges': 'bytes', 'X-Storage-Policy': 'Policy-0', 'Date': 'Fri, 16 Dec 2016 18:29:05 GMT', 'X-Timestamp': '1481912480.41664', 'X-Trans-Id': 'tx60ec128d9dbf44b9add68-0058543271dfw1', 'X-Container-Bytes-Used': '0', 'Content-Type': 'text/plain; charset=utf-8'})])
    container = self.cloud.create_container(self.container)
    self.assert_calls()
    self.assertIsNotNone(container)