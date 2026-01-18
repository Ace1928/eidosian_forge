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
def test_before_request_functions_must_be_added(self):

    class TestAPINoBefore(_TestRestfulAPI):

        def _register_before_request_functions(self, functions=None):
            pass
    self.assertRaises(AssertionError, TestAPINoBefore)