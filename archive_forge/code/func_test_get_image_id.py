import os
from unittest import mock
import glance_store
from oslo_config import cfg
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context
from glance import housekeeping
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_image_id(self):
    self.assertEqual(uuids.some_random_uuid, self.cleaner.get_image_id(uuids.some_random_uuid))
    self.assertEqual(uuids.some_random_uuid, self.cleaner.get_image_id('%s.qcow2' % uuids.some_random_uuid))
    self.assertEqual(uuids.some_random_uuid, self.cleaner.get_image_id('%s.uc' % uuids.some_random_uuid))
    self.assertEqual(uuids.some_random_uuid, self.cleaner.get_image_id('%s.blah' % uuids.some_random_uuid))
    self.assertIsNone(self.cleaner.get_image_id('foo'))
    self.assertIsNone(self.cleaner.get_image_id('foo.bar'))