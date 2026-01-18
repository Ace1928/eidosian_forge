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
def test_list_images_paginated(self):
    marker = str(uuid.uuid4())
    self.register_uris([dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2'), json={'images': [self.fake_image_dict], 'next': '/v2/images?marker={marker}'.format(marker=marker)}), dict(method='GET', uri=self.get_mock_url('image', append=['images'], base_url_append='v2', qs_elements=['marker={marker}'.format(marker=marker)]), json=self.fake_search_return)])
    [self._compare_images(b, a) for a, b in zip(self.cloud.list_images(), [self.fake_image_dict, self.fake_image_dict])]
    self.assert_calls()