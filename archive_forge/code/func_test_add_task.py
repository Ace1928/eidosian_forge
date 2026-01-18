import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_add_task(self):
    task_type = 'import'
    image_id = 'fake_image_id'
    user_id = 'fake_user'
    request_id = 'fake_request_id'
    task = self.task_factory.new_task(task_type, None, image_id, user_id, request_id, task_input=self.fake_task_input)
    self.assertEqual(task.updated_at, task.created_at)
    self.task_repo.add(task)
    retrieved_task = self.task_repo.get(task.task_id)
    self.assertEqual(task.updated_at, retrieved_task.updated_at)
    self.assertEqual(self.fake_task_input, retrieved_task.task_input)
    self.assertEqual(image_id, task.image_id)
    self.assertEqual(user_id, task.user_id)
    self.assertEqual(request_id, task.request_id)