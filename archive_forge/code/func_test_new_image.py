import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_new_image(self):
    image = self.image_factory.new_image(image_id=UUID1, name='image-1', min_disk=256, owner=TENANT1)
    self.assertEqual(UUID1, image.image_id)
    self.assertIsNotNone(image.created_at)
    self.assertEqual(image.created_at, image.updated_at)
    self.assertEqual('queued', image.status)
    self.assertEqual('shared', image.visibility)
    self.assertEqual(TENANT1, image.owner)
    self.assertEqual('image-1', image.name)
    self.assertIsNone(image.size)
    self.assertEqual(256, image.min_disk)
    self.assertEqual(0, image.min_ram)
    self.assertFalse(image.protected)
    self.assertIsNone(image.disk_format)
    self.assertIsNone(image.container_format)
    self.assertEqual({}, image.extra_properties)
    self.assertEqual(set([]), image.tags)