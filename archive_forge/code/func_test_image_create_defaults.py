import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
@mock.patch.object(timeutils, 'utcnow')
def test_image_create_defaults(self, mock_utcnow):
    mock_utcnow.return_value = datetime.datetime.utcnow()
    create_time = timeutils.utcnow()
    values = {'status': 'queued', 'created_at': create_time, 'updated_at': create_time}
    image = self.db_api.image_create(self.context, values)
    self.assertIsNone(image['name'])
    self.assertIsNone(image['container_format'])
    self.assertEqual(0, image['min_ram'])
    self.assertEqual(0, image['min_disk'])
    self.assertIsNone(image['owner'])
    self.assertEqual('shared', image['visibility'])
    self.assertIsNone(image['size'])
    self.assertIsNone(image['checksum'])
    self.assertIsNone(image['disk_format'])
    self.assertEqual([], image['locations'])
    self.assertFalse(image['protected'])
    self.assertFalse(image['deleted'])
    self.assertIsNone(image['deleted_at'])
    self.assertEqual([], image['properties'])
    self.assertEqual(create_time, image['created_at'])
    self.assertEqual(create_time, image['updated_at'])
    self.assertTrue(uuid.UUID(image['id']))
    self.assertNotIn('tags', image)