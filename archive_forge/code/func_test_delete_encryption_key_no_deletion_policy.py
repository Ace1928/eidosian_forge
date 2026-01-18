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
def test_delete_encryption_key_no_deletion_policy(self):
    request = unit_test_utils.get_fake_request()
    fake_encryption_key = self.controller._key_manager.store(request.context, mock.Mock())
    props = {'cinder_encryption_key_id': fake_encryption_key}
    image = _domain_fixture(UUID2, name='image-2', owner=TENANT2, checksum='ca425b88f047ce8ec45ee90e813ada91', os_hash_algo=FAKEHASHALGO, os_hash_value=MULTIHASH1, created_at=DATETIME, updated_at=DATETIME, size=1024, virtual_size=3072, extra_properties=props)
    self.controller._delete_encryption_key(request.context, image)
    key = self.controller._key_manager.get(request.context, fake_encryption_key)
    self.assertEqual(fake_encryption_key, key._id)