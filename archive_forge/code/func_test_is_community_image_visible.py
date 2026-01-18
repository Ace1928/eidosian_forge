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
def test_is_community_image_visible(self):
    TENANT1 = str(uuid.uuid4())
    TENANT2 = str(uuid.uuid4())
    owners_ctxt = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    viewing_ctxt = context.RequestContext(is_admin=False, user=TENANT2, auth_token='user:%s:user' % TENANT2)
    UUIDX = str(uuid.uuid4())
    image = self.db_api.image_create(owners_ctxt, {'id': UUIDX, 'status': 'queued', 'visibility': 'community', 'owner': TENANT1})
    result = self.db_api.is_image_visible(owners_ctxt, image)
    self.assertTrue(result)
    result = self.db_api.is_image_visible(viewing_ctxt, image)
    self.assertTrue(result)