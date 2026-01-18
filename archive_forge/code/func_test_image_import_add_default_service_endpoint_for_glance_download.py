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
@mock.patch.object(glance.domain.TaskFactory, 'new_task')
@mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
def test_image_import_add_default_service_endpoint_for_glance_download(self, mock_get, mock_nt):
    request = unit_test_utils.get_fake_request()
    mock_get.return_value = FakeImage(status='queued')
    body = {'method': {'name': 'glance-download', 'glance_image_id': UUID4, 'glance_region': 'REGION2'}}
    self.controller.import_image(request, UUID4, body)
    expected_req = {'method': {'name': 'glance-download', 'glance_image_id': UUID4, 'glance_region': 'REGION2', 'glance_service_interface': 'public'}}
    self.assertEqual(expected_req, mock_nt.call_args.kwargs['task_input']['import_req'])