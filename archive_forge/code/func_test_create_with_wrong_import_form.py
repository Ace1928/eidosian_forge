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
def test_create_with_wrong_import_form(self):
    request = unit_test_utils.get_fake_request()
    wrong_import_from = ['swift://cloud.foo/myaccount/mycontainer/path', 'file:///path', 'cinder://volume-id']
    executor_factory = self.gateway.get_task_executor_factory(request.context)
    task_repo = self.gateway.get_task_repo(request.context)
    for import_from in wrong_import_from:
        task = {'type': 'import', 'input': {'import_from': import_from, 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'qcow2', 'container_format': 'bare', 'name': 'test-task'}}}
        new_task = self.controller.create(request, task=task)
        task_executor = executor_factory.new_task_executor(request.context)
        task_executor.begin_processing(new_task.task_id)
        final_task = task_repo.get(new_task.task_id)
        self.assertEqual('failure', final_task.status)
        if import_from.startswith('file:///'):
            msg = 'File based imports are not allowed. Please use a non-local source of image data.'
        else:
            supported = ['http']
            msg = 'The given uri is not valid. Please specify a valid uri from the following list of supported uri %(supported)s' % {'supported': supported}
        self.assertEqual(msg, final_task.message)