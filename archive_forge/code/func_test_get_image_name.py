from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_image_name(self):
    self.use_glance()
    image_id = self.getUniqueString()
    fake_image = fakes.make_fake_image(image_id=image_id)
    list_return = {'images': [fake_image]}
    self.register_uris([dict(method='GET', uri='https://image.example.com/v2/images', json=list_return), dict(method='GET', uri='https://image.example.com/v2/images', json=list_return)])
    self.assertEqual('fake_image', self.cloud.get_image_name(image_id))
    self.assertEqual('fake_image', self.cloud.get_image_name('fake_image'))
    self.assert_calls()