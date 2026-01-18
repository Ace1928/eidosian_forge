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
def test_list_images_no_filter_deleted(self):
    deleted_image = self.fake_image_dict.copy()
    deleted_image['status'] = 'deleted'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json={'images': [self.fake_image_dict, deleted_image]})])
    [self._compare_images(a, b) for a, b in zip([self.fake_image_dict], self.cloud.list_images(filter_deleted=False))]
    self.assert_calls()