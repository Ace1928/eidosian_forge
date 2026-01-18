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
def test_get_object_segment_size_http_412(self):
    self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', status_code=412, reason='Precondition failed')])
    self.assertEqual(_proxy.DEFAULT_OBJECT_SEGMENT_SIZE, self.cloud.get_object_segment_size(None))
    self.assert_calls()