import datetime
import hashlib
import http.client as http
import os
import requests
from unittest import mock
import uuid
from castellan.common import exception as castellan_exception
import glance_store as store
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import fixture
import testtools
import webob
import webob.exc
import glance.api.v2.image_actions
import glance.api.v2.images
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance import domain
import glance.notifier
import glance.schema
from glance.tests.unit import base
from glance.tests.unit.keymgr import fake as fake_keymgr
import glance.tests.unit.utils as unit_test_utils
from glance.tests.unit.v2 import test_tasks_resource
import glance.tests.utils as test_utils
@mock.patch('glance.db.simple.api.image_set_property_atomic')
@mock.patch('glance.context.RequestContext.elevated')
@mock.patch.object(glance.domain.TaskFactory, 'new_task')
@mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
def test_image_import_copy_allowed_by_policy(self, mock_get, mock_new_task, mock_elevated, mock_spa, allowed=True):
    request = unit_test_utils.get_fake_request(tenant=TENANT2)
    mock_get.return_value = FakeImage(status='active', locations=[])
    self.policy.rules = {'copy_image': allowed}
    req_body = {'method': {'name': 'copy-image'}, 'stores': ['cheap']}
    with mock.patch.object(self.controller.gateway, 'get_task_executor_factory', side_effect=self.controller.gateway.get_task_executor_factory) as mock_tef:
        self.controller.import_image(request, UUID4, req_body)
        mock_tef.assert_called_once_with(request.context, admin_context=mock_elevated.return_value)
    expected_input = {'image_id': UUID4, 'import_req': mock.ANY, 'backend': mock.ANY}
    mock_new_task.assert_called_with(task_type='api_image_import', owner=TENANT2, task_input=expected_input, image_id=UUID4, user_id=request.context.user_id, request_id=request.context.request_id)