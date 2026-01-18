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
def test_is_image_visible(self):
    TENANT1 = str(uuid.uuid4())
    TENANT2 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
    UUIDX = str(uuid.uuid4())
    image = self.db_api.image_create(ctxt1, {'id': UUIDX, 'status': 'queued', 'is_public': False, 'owner': TENANT1})
    values = {'image_id': UUIDX, 'member': TENANT2, 'can_share': False}
    self.db_api.image_member_create(ctxt1, values)
    result = self.db_api.is_image_visible(ctxt2, image)
    self.assertTrue(result)
    members = self.db_api.image_member_find(ctxt1, image_id=UUIDX)
    self.db_api.image_member_delete(ctxt1, members[0]['id'])
    result = self.db_api.is_image_visible(ctxt2, image)
    self.assertFalse(result)