import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_image_stage_raises_internal_error(self):
    image_id = str(uuid.uuid4())
    request = unit_test_utils.get_fake_request()
    self.image_repo.result = exception.ServerError()
    self.assertRaises(exception.ServerError, self.controller.stage, request, image_id, 'YYYYYYY', 7)