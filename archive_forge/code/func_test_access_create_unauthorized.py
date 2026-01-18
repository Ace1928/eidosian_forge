import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_access_create_unauthorized(self):
    """Disallow access with bad request, fail with 403"""
    rules = {'tasks_api_access': False, 'add_task': True}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request()
    request.body = jsonutils.dump_as_bytes(self.bad_task)
    self.assertRaises(webob.exc.HTTPForbidden, self.deserializer.create, request)