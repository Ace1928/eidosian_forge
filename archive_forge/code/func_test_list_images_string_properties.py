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
def test_list_images_string_properties(self):
    image_dict = self.fake_image_dict.copy()
    image_dict['properties'] = 'list,of,properties'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json={'images': [image_dict]})])
    images = self.cloud.list_images()
    [self._compare_images(a, b) for a, b in zip([image_dict], images)]
    self.assertEqual(images[0]['properties']['properties'], 'list,of,properties')
    self.assert_calls()