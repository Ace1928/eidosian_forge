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
def test_create_ensure_expires_at_is_not_returned(self):
    response = webob.Response()
    self.serializer.create(response, self.fixtures[0])
    serialized_task = jsonutils.loads(response.body)
    self.assertEqual(http.CREATED, response.status_int)
    self.assertEqual(self.fixtures[0].task_id, serialized_task['id'])
    self.assertEqual(self.fixtures[0].task_input, serialized_task['input'])
    self.assertNotIn('expires_at', serialized_task)
    self.assertEqual('application/json', response.content_type)
    response = webob.Response()
    self.serializer.create(response, self.fixtures[1])
    serialized_task = jsonutils.loads(response.body)
    self.assertEqual(http.CREATED, response.status_int)
    self.assertEqual(self.fixtures[1].task_id, serialized_task['id'])
    self.assertEqual(self.fixtures[1].task_input, serialized_task['input'])
    self.assertNotIn('expires_at', serialized_task)
    self.assertEqual('application/json', response.content_type)