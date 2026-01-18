import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_new_task(self):
    task_type = 'import'
    owner = TENANT1
    task_input = 'input'
    image_id = 'fake_image_id'
    user_id = 'fake_user'
    request_id = 'fake_request_id'
    task = self.task_factory.new_task(task_type, owner, image_id, user_id, request_id, task_input=task_input, result='test_result', message='test_message')
    self.assertIsNotNone(task.task_id)
    self.assertIsNotNone(task.created_at)
    self.assertEqual(task_type, task.type)
    self.assertEqual(task.created_at, task.updated_at)
    self.assertEqual('pending', task.status)
    self.assertIsNone(task.expires_at)
    self.assertEqual(owner, task.owner)
    self.assertEqual(task_input, task.task_input)
    self.assertEqual('test_message', task.message)
    self.assertEqual('test_result', task.result)
    self.assertEqual(image_id, task.image_id)
    self.assertEqual(user_id, task.user_id)
    self.assertEqual(request_id, task.request_id)