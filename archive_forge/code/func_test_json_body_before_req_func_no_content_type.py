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
def test_json_body_before_req_func_no_content_type(self):
    with self.test_request_context(data='{"key": "value"}'):
        json_body.json_body_before_request()
    with self.test_request_context(headers={'Content-Type': ''}, data='{"key": "value"}'):
        json_body.json_body_before_request()