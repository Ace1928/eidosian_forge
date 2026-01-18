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
@mock.patch.object(glance.quota, '_calc_required_size')
@mock.patch.object(glance.location, '_check_image_location')
@mock.patch.object(glance.location.ImageRepoProxy, '_set_acls')
@mock.patch.object(store, 'get_size_from_uri_and_backend')
@mock.patch.object(store, 'get_size_from_backend')
def test_add_location_invalid_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
    mock_calc.return_value = 1
    mock_get_size.return_value = 1
    mock_get_size_uri.return_value = 1
    self.config(show_multiple_locations=True)
    image_id = str(uuid.uuid4())
    self.images = [_db_fixture(image_id, owner=TENANT1, checksum=None, os_hash_algo=None, os_hash_value=None, name='1', disk_format='raw', container_format='bare', status='queued')]
    self.db.image_create(None, self.images[0])
    request = unit_test_utils.get_fake_request()
    location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {}}
    changes = [{'op': 'add', 'path': ['locations', '-'], 'value': location}]
    changes[0]['value']['validation_data'] = {'checksum': 'something the same length as md5', 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}
    self.assertRaisesRegex(webob.exc.HTTPConflict, 'checksum .* is not a valid hexadecimal value', self.controller.update, request, image_id, changes)
    changes[0]['value']['validation_data'] = {'checksum': '0123456789abcdef', 'os_hash_algo': 'sha512', 'os_hash_value': MULTIHASH1}
    self.assertRaisesRegex(webob.exc.HTTPConflict, 'checksum .* is not the correct size', self.controller.update, request, image_id, changes)
    changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha256', 'os_hash_value': MULTIHASH1}
    self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_algo must be sha512', self.controller.update, request, image_id, changes)
    changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': 'not a hex value'}
    self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_value .* is not a valid hexadecimal value', self.controller.update, request, image_id, changes)
    changes[0]['value']['validation_data'] = {'checksum': CHKSUM, 'os_hash_algo': 'sha512', 'os_hash_value': '0123456789abcdef'}
    self.assertRaisesRegex(webob.exc.HTTPConflict, 'os_hash_value .* is not the correct size for sha512', self.controller.update, request, image_id, changes)