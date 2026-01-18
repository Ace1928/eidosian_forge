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
def test_get_object_segment_size_below_min(self):
    self.register_uris([dict(method='GET', uri='https://object-store.example.com/info', json=dict(swift={'max_file_size': 1000}, slo={'min_segment_size': 500}), headers={'Content-Type': 'application/json'})])
    self.assertEqual(500, self.cloud.get_object_segment_size(400))
    self.assertEqual(900, self.cloud.get_object_segment_size(900))
    self.assertEqual(1000, self.cloud.get_object_segment_size(1000))
    self.assertEqual(1000, self.cloud.get_object_segment_size(1100))