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
def test_db_purge(self):
    self.db_api.purge_deleted_rows(self.adm_context, 1, 5)
    images = self.db_api.image_get_all(self.adm_context)
    self.assertEqual(len(images), 3)
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(len(tasks), 2)