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
def test_image_member_count(self):
    TENANT1 = str(uuid.uuid4())
    self.db_api.image_member_create(self.context, {'member': TENANT1, 'image_id': UUID1})
    actual = self.db_api.image_member_count(self.context, UUID1)
    self.assertEqual(1, actual)