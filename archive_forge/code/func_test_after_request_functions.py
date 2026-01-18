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
def test_after_request_functions(self):
    attr = uuid.uuid4().hex

    def do_something(resp):
        setattr(flask.g, attr, True)
        resp.status_code = 420
        return resp

    class TestAPI(_TestRestfulAPI):

        def _register_after_request_functions(self, functions=None):
            functions = functions or []
            functions.append(do_something)
            super(TestAPI, self)._register_after_request_functions(functions)
    api = TestAPI(resources=[_TestResourceWithCollectionInfo])
    self.public_app.app.register_blueprint(api.blueprint)
    token = self._get_token()
    with self.test_client() as c:
        c.get('/v3/arguments', headers={'X-Auth-Token': token}, expected_status_code=420)