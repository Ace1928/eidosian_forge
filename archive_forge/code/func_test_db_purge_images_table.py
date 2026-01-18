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
def test_db_purge_images_table(self):
    images = self.db_api.image_get_all(self.adm_context)
    self.assertEqual(len(images), 3)
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(len(tasks), 3)
    for image in images:
        session = self.db_api.get_session()
        with session.begin():
            session.execute(sql.delete(models.ImageLocation).where(models.ImageLocation.image_id == image['id']))
    self.db_api.purge_deleted_rows(self.adm_context, 1, 5)
    self.db_api.purge_deleted_rows_from_images(self.adm_context, 1, 5)
    images = self.db_api.image_get_all(self.adm_context)
    self.assertEqual(len(images), 2)
    tasks = self.db_api.task_get_all(self.adm_context)
    self.assertEqual(len(tasks), 2)