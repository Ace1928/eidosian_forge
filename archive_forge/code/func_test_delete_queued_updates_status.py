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
def test_delete_queued_updates_status(self):
    """Ensure status of queued image is updated (LP bug #1048851)"""
    request = unit_test_utils.get_fake_request(is_admin=True)
    image = self.db.image_create(request.context, {'status': 'queued'})
    image_id = image['id']
    self.controller.delete(request, image_id)
    image = self.db.image_get(request.context, image_id, force_show_deleted=True)
    self.assertTrue(image['deleted'])
    self.assertEqual('deleted', image['status'])