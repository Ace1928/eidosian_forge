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
def test_image_get_not_owned(self):
    TENANT1 = str(uuid.uuid4())
    TENANT2 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
    image = self.db_api.image_create(ctxt1, {'status': 'queued', 'owner': TENANT1})
    self.assertRaises(exception.Forbidden, self.db_api.image_get, ctxt2, image['id'])