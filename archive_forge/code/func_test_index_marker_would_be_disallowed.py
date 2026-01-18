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
def test_index_marker_would_be_disallowed(self):
    self.config(limit_param_default=1, api_limit_max=10)
    request = unit_test_utils.get_fake_request(is_admin=True)

    def fake_enforce(context, action, target=None, **kw):
        assert target is not None
        if target['project_id'] != TENANT1:
            raise exception.Forbidden()
    output = self.controller.index(request, sort_dir=['asc'], limit=3)
    self.assertEqual(UUID3, output['next_marker'])
    self.assertEqual(3, len(output['images']))
    with mock.patch.object(self.controller.policy, 'enforce', new=fake_enforce):
        output = self.controller.index(request, sort_dir=['asc'], limit=3)
    self.assertEqual(UUID2, output['next_marker'])
    self.assertEqual(2, len(output['images']))