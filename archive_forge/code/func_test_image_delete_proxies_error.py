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
def test_image_delete_proxies_error(self, mock_client):
    self.config(worker_self_reference_url='http://glance-worker2.openstack.org')
    request = unit_test_utils.get_fake_request('/v2/images/%s' % UUID4, method='DELETE')
    with mock.patch.object(glance.notifier.ImageRepoProxy, 'get') as mock_get:
        mock_get.return_value = FakeImage(status='uploading')
        mock_get.return_value.extra_properties['os_glance_stage_host'] = 'https://glance-worker1.openstack.org'
        remote_hdrs = {'x-openstack-request-id': 'remote-req'}
        mock_resp = mock.MagicMock(location='/target', status_code=456, reason='No thanks', headers=remote_hdrs)
        mock_client.return_value.delete.return_value = mock_resp
        exc = self.assertRaises(webob.exc.HTTPError, self.controller.delete, request, UUID4)
        self.assertEqual('456 No thanks', exc.status)
        mock_client.return_value.delete.assert_called_once_with('https://glance-worker1.openstack.org/v2/images/%s' % UUID4, json=None, timeout=60)