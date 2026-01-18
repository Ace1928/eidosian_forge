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
def test_admin_as_user_true(self):
    images = self.db_api.image_get_all(self.admin_context, admin_as_user=True)
    self.assertEqual(7, len(images))
    for i in images:
        self.assertTrue('public' == i['visibility'] or i['owner'] == self.admin_tenant)