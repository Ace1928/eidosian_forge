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
def test_get_ensure_expires_at_not_returned(self):
    expected = {'id': UUID1, 'type': 'import', 'status': 'pending', 'input': {'loc': 'fake'}, 'result': {}, 'owner': TENANT1, 'message': '', 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID1, 'schema': '/v2/schemas/task', 'image_id': 'fake_image_id', 'user_id': 'fake_user', 'request_id': 'fake_request_id'}
    response = webob.Response()
    self.serializer.get(response, self.fixtures[0])
    actual = jsonutils.loads(response.body)
    self.assertEqual(expected, actual)
    self.assertEqual('application/json', response.content_type)
    expected = {'id': UUID2, 'type': 'import', 'status': 'processing', 'input': {'loc': 'bake'}, 'result': {}, 'owner': TENANT2, 'message': '', 'created_at': ISOTIME, 'updated_at': ISOTIME, 'self': '/v2/tasks/%s' % UUID2, 'schema': '/v2/schemas/task', 'image_id': 'fake_image_id', 'user_id': 'fake_user', 'request_id': 'fake_request_id'}
    response = webob.Response()
    self.serializer.get(response, self.fixtures[1])
    actual = jsonutils.loads(response.body)
    self.assertEqual(expected, actual)
    self.assertEqual('application/json', response.content_type)