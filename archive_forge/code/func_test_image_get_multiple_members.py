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
def test_image_get_multiple_members(self):
    TENANT1 = str(uuid.uuid4())
    TENANT2 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
    UUIDX = str(uuid.uuid4())
    self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'is_public': False, 'owner': TENANT1})
    values = {'image_id': UUIDX, 'member': TENANT2, 'can_share': False}
    self.db_api.image_member_create(ctxt1, values)
    image = self.db_api.image_get(ctxt2, UUIDX)
    self.assertEqual(UUIDX, image['id'])
    images = self.db_api.image_get_all(ctxt2)
    self.assertEqual(3, len(images))
    images = self.db_api.image_get_all(ctxt2, member_status='rejected')
    self.assertEqual(3, len(images))
    images = self.db_api.image_get_all(ctxt2, filters={'visibility': 'shared'})
    self.assertEqual(0, len(images))
    images = self.db_api.image_get_all(ctxt2, member_status='pending', filters={'visibility': 'shared'})
    self.assertEqual(1, len(images))
    images = self.db_api.image_get_all(ctxt2, member_status='all', filters={'visibility': 'shared'})
    self.assertEqual(1, len(images))
    images = self.db_api.image_get_all(ctxt2, member_status='pending')
    self.assertEqual(4, len(images))
    images = self.db_api.image_get_all(ctxt2, member_status='all')
    self.assertEqual(4, len(images))