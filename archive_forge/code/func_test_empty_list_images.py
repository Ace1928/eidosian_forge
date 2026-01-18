import io
import operator
import tempfile
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack import connection
from openstack import exceptions
from openstack.image.v1 import image as image_v1
from openstack.image.v2 import image
from openstack.tests import fakes
from openstack.tests.unit import base
def test_empty_list_images(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json={'images': []})])
    self.assertEqual([], self.cloud.list_images())
    self.assert_calls()