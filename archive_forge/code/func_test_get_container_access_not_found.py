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
def test_get_container_access_not_found(self):
    self.register_uris([dict(method='HEAD', uri=self.container_endpoint, status_code=404)])
    with testtools.ExpectedException(exceptions.SDKException, 'Container not found: %s' % self.container):
        self.cloud.get_container_access(self.container)