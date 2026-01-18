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
def test_mapped_resource_routes(self):

    class MappedResource(flask_restful.Resource):

        def post(self):
            rbac_enforcer.enforcer.RBACEnforcer().enforce_call(action='example:allowed')
            post_body = flask.request.get_json()
            return ({'post_body': post_body}, 201)
    resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
    restful_api = _TestRestfulAPI(resource_mapping=[resource_map], resources=[])
    self.public_app.app.register_blueprint(restful_api.blueprint)
    token = self._get_token()
    with self.test_client() as c:
        body = {'test_value': uuid.uuid4().hex}
        resp = c.post('/v3/test_api', json=body, headers={'X-Auth-Token': token})
        self.assertEqual(body, resp.json['post_body'])