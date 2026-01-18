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
def test_add_location_same_validation_data(self, mock_get_size, mock_get_size_uri, mock_set_acls, mock_check_loc, mock_calc):
    mock_calc.return_value = 1
    mock_get_size.return_value = 1
    mock_get_size_uri.return_value = 1
    self.config(show_multiple_locations=True)
    image_id = str(uuid.uuid4())
    os_hash_value = '6513f21e44aa3da349f248188a44bc304a3653a04122d8fb4535423c8e1d14cd6a153f735bb0982e2161b5b5186106570c17a9e58b64dd39390617cd5a350f78'
    self.images = [_db_fixture(image_id, owner=TENANT1, name='1', disk_format='raw', container_format='bare', status='active', checksum='checksum1', os_hash_algo='sha512', os_hash_value=os_hash_value)]
    self.db.image_create(None, self.images[0])
    request = unit_test_utils.get_fake_request()
    new_location = {'url': '%s/fake_location_1' % BASE_URI, 'metadata': {}, 'validation_data': {'checksum': 'checksum1', 'os_hash_algo': 'sha512', 'os_hash_value': os_hash_value}}
    changes = [{'op': 'add', 'path': ['locations', '-'], 'value': new_location}]
    output = self.controller.update(request, image_id, changes)
    self.assertEqual(image_id, output.image_id)
    self.assertEqual(1, len(output.locations))
    self.assertEqual(new_location, output.locations[0])
    self.assertEqual('active', output.status)