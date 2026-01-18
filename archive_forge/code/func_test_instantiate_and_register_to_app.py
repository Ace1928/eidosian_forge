import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
def test_instantiate_and_register_to_app(self):
    self.restful_api_opts = {}
    self.restful_api = _TestRestfulAPI.instantiate_and_register_to_app(self.public_app.app)
    self.cleanup_instance('restful_api_opts')
    self.cleanup_instance('restful_api')
    self._make_requests()