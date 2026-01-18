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
def test_image_member_update_status(self):
    TENANT1 = str(uuid.uuid4())
    self.context.auth_token = 'user:%s:user' % TENANT1
    member = self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
    member_id = member.pop('id')
    member.pop('created_at')
    member.pop('updated_at')
    expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'pending', 'can_share': False, 'deleted': False}
    self.assertEqual(expected, member)
    self.delay_inaccurate_clock()
    member = self.db_api.image_member_update(self.context, member_id, {'status': 'accepted'})
    self.assertNotEqual(member['created_at'], member['updated_at'])
    member.pop('id')
    member.pop('created_at')
    member.pop('updated_at')
    expected = {'member': TENANT1, 'image_id': UUID1, 'status': 'accepted', 'can_share': False, 'deleted': False}
    self.assertEqual(expected, member)
    members = self.db_api.image_member_find(self.context, member=TENANT1, image_id=UUID1)
    member = members[0]
    member.pop('id')
    member.pop('created_at')
    member.pop('updated_at')
    self.assertEqual(expected, member)