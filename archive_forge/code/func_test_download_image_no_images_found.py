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
def test_download_image_no_images_found(self):
    self.register_uris([dict(method='GET', uri='https://image.example.com/v2/images/{name}'.format(name=self.image_name), status_code=404), dict(method='GET', uri='https://image.example.com/v2/images?name={name}'.format(name=self.image_name), json=dict(images=[])), dict(method='GET', uri='https://image.example.com/v2/images?os_hidden=True', json=dict(images=[]))])
    self.assertRaises(exceptions.NotFoundException, self.cloud.download_image, self.image_name, output_path='fake_path')
    self.assert_calls()