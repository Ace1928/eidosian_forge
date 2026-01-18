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
def test_delete_container(self):
    self.register_uris([dict(method='DELETE', uri=self.container_endpoint)])
    self.assertTrue(self.cloud.delete_container(self.container))
    self.assert_calls()