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
def test_cleanup_stale_task_progress(self):
    img_repo = mock.MagicMock()
    image = mock.MagicMock()
    task = mock.MagicMock()
    task.task_input = {}
    image.extra_properties = {}
    self.controller._cleanup_stale_task_progress(img_repo, image, task)
    img_repo.save.assert_not_called()
    task.task_input = {'backend': []}
    self.controller._cleanup_stale_task_progress(img_repo, image, task)
    img_repo.save.assert_not_called()
    task.task_input = {'backend': ['store1', 'store2']}
    self.controller._cleanup_stale_task_progress(img_repo, image, task)
    img_repo.save.assert_not_called()
    image.extra_properties = {'os_glance_failed_import': 'store3'}
    self.controller._cleanup_stale_task_progress(img_repo, image, task)
    img_repo.save.assert_not_called()
    image.extra_properties = {'os_glance_importing_to_stores': 'foo,store1,bar', 'os_glance_failed_import': 'foo,store2,bar'}
    self.controller._cleanup_stale_task_progress(img_repo, image, task)
    img_repo.save.assert_called_once_with(image)
    self.assertEqual({'os_glance_importing_to_stores': 'foo,bar', 'os_glance_failed_import': 'foo,bar'}, image.extra_properties)