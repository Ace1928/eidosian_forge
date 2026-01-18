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
def test_index_store_multiple_location_enabled(self):
    self.config(show_multiple_locations=True)
    request = webob.Request.blank('/v2/images')
    response = webob.Response(request=request)
    (self.serializer.index(response, {'images': [self.location_data_image]}),)
    images = jsonutils.loads(response.body)['images']
    location = images[0]['locations'][0]
    self.assertEqual(location['url'], self.location_data_image_url)
    self.assertEqual(location['metadata'], self.location_data_image_meta)