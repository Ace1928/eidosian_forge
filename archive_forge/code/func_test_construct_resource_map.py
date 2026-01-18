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
def test_construct_resource_map(self):
    resource_name = 'arguments'
    param_relation = json_home.build_v3_parameter_relation('argument_id')
    alt_rel_func = functools.partial(json_home.build_v3_extension_resource_relation, extension_name='extension', extension_version='1.0')
    url = '/v3/arguments/<string:argument_id>'
    old_url = [dict(url='/v3/old_arguments/<string:argument_id>', json_home=flask_common.construct_json_home_data(rel='arguments', resource_relation_func=alt_rel_func))]
    mapping = flask_common.construct_resource_map(resource=_TestResourceWithCollectionInfo, url=url, resource_kwargs={}, alternate_urls=old_url, rel=resource_name, status=json_home.Status.EXPERIMENTAL, path_vars={'argument_id': param_relation}, resource_relation_func=json_home.build_v3_resource_relation)
    self.assertEqual(_TestResourceWithCollectionInfo, mapping.resource)
    self.assertEqual(url, mapping.url)
    self.assertEqual(json_home.build_v3_resource_relation(resource_name), mapping.json_home_data.rel)
    self.assertEqual(json_home.Status.EXPERIMENTAL, mapping.json_home_data.status)
    self.assertEqual({'argument_id': param_relation}, mapping.json_home_data.path_vars)
    self.assertEqual(1, len(mapping.alternate_urls))
    alt_url_data = mapping.alternate_urls[0]
    self.assertEqual(old_url[0]['url'], alt_url_data['url'])
    self.assertEqual(old_url[0]['json_home'], alt_url_data['json_home'])