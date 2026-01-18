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
def test_image_destroy_with_delete_all(self):
    """Check the image child element's _image_delete_all methods.

        checks if all the image_delete_all methods deletes only the child
        elements of the image to be deleted.
        """
    TENANT2 = str(uuid.uuid4())
    location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]

    def _create_image_with_child_entries():
        fixture = {'status': 'queued', 'locations': location_data}
        image_id = self.db_api.image_create(self.context, fixture)['id']
        fixture = {'name': 'ping', 'value': 'pong', 'image_id': image_id}
        self.db_api.image_property_create(self.context, fixture)
        fixture = {'image_id': image_id, 'member': TENANT2, 'can_share': False}
        self.db_api.image_member_create(self.context, fixture)
        self.db_api.image_tag_create(self.context, image_id, 'snarf')
        return image_id
    ACTIVE_IMG_ID = _create_image_with_child_entries()
    DEL_IMG_ID = _create_image_with_child_entries()
    deleted_image = self.db_api.image_destroy(self.adm_context, DEL_IMG_ID)
    self.assertTrue(deleted_image['deleted'])
    self.assertTrue(deleted_image['deleted_at'])
    self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, DEL_IMG_ID)
    active_image = self.db_api.image_get(self.context, ACTIVE_IMG_ID)
    self.assertFalse(active_image['deleted'])
    self.assertFalse(active_image['deleted_at'])
    self.assertEqual(2, len(active_image['locations']))
    self.assertIn('id', active_image['locations'][0])
    self.assertIn('id', active_image['locations'][1])
    active_image['locations'][0].pop('id')
    active_image['locations'][1].pop('id')
    self.assertEqual(location_data, active_image['locations'])
    self.assertEqual(1, len(active_image['properties']))
    prop = active_image['properties'][0]
    self.assertEqual(('ping', 'pong', ACTIVE_IMG_ID), (prop['name'], prop['value'], prop['image_id']))
    self.assertEqual((False, None), (prop['deleted'], prop['deleted_at']))
    self.context.auth_token = 'user:%s:user' % TENANT2
    members = self.db_api.image_member_find(self.context, ACTIVE_IMG_ID)
    self.assertEqual(1, len(members))
    member = members[0]
    self.assertEqual((TENANT2, ACTIVE_IMG_ID, False), (member['member'], member['image_id'], member['can_share']))
    tags = self.db_api.image_tag_get_all(self.context, ACTIVE_IMG_ID)
    self.assertEqual(['snarf'], tags)