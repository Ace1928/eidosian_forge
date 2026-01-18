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
def test_new_task_invalid_type(self):
    task_type = 'blah'
    image_id = 'fake_image_id'
    user_id = 'fake_user'
    request_id = 'fake_request_id'
    owner = TENANT1
    self.assertRaises(exception.InvalidTaskType, self.task_factory.new_task, task_type, owner, image_id, user_id, request_id)