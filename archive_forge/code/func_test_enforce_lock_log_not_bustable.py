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
def test_enforce_lock_log_not_bustable(self, task_status='processing'):
    task = test_tasks_resource._db_fixture(test_tasks_resource.UUID1, status=task_status)
    self.db.task_create(None, task)
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    image = FakeImage()
    image.extra_properties['os_glance_import_task'] = task['id']
    time_fixture = fixture.TimeFixture(task['updated_at'] + datetime.timedelta(minutes=55))
    self.useFixture(time_fixture)
    expected_expire = 300
    if task_status == 'pending':
        expected_expire += 3600
    with mock.patch.object(glance.api.v2.images, 'LOG') as mock_log:
        self.assertRaises(exception.Conflict, self.controller._enforce_import_lock, request, image)
        mock_log.warning.assert_called_once_with('Image %(image)s has active import task %(task)s in status %(status)s; lock remains valid for %(expire)i more seconds', {'image': image.id, 'task': task['id'], 'status': task_status, 'expire': expected_expire})