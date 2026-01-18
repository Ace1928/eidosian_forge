import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_create_with_properties_missed(self):
    request = unit_test_utils.get_fake_request()
    executor_factory = self.gateway.get_task_executor_factory(request.context)
    task_repo = self.gateway.get_task_repo(request.context)
    task = {'type': 'import', 'input': {'import_from': 'swift://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2'}}
    new_task = self.controller.create(request, task=task)
    task_executor = executor_factory.new_task_executor(request.context)
    task_executor.begin_processing(new_task.task_id)
    final_task = task_repo.get(new_task.task_id)
    self.assertEqual('failure', final_task.status)
    msg = "Input does not contain 'image_properties' field"
    self.assertEqual(msg, final_task.message)