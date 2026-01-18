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
@mock.patch('glance.context.get_ksa_client')
@mock.patch.object(glance.notifier.ImageRepoProxy, 'get')
@mock.patch.object(glance.notifier.ImageRepoProxy, 'remove')
def test_image_delete_deletes_locally_on_error(self, mock_remove, mock_get, mock_client):
    self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
    request = unit_test_utils.get_fake_request('/v2/images/%s' % UUID4, method='DELETE')
    image = FakeImage(status='uploading')
    mock_get.return_value = image
    image.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
    image.delete = mock.MagicMock()
    mock_client.return_value.delete.side_effect = requests.exceptions.ConnectTimeout
    self.controller.delete(request, UUID4)
    mock_get.return_value.delete.assert_called_once_with()
    mock_remove.assert_called_once_with(image)
    mock_client.return_value.delete.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s' % UUID4, json=None, timeout=60)