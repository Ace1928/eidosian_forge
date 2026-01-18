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
def test_HTTP_OPTIONS_is_unenforced(self):

    class MappedResource(flask_restful.Resource):

        def post(self):
            pass
    resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
    restful_api = _TestRestfulAPI(resource_mapping=[resource_map], resources=[])
    self.public_app.app.register_blueprint(restful_api.blueprint)
    with self.test_client() as c:
        r = c.options('/v3/test_api')
        self.assertEqual(set(['OPTIONS', 'POST']), set([v.lstrip().rstrip() for v in r.headers['Allow'].split(',')]))
        self.assertEqual(r.headers['Content-Length'], '0')
        self.assertEqual(r.data, b'')