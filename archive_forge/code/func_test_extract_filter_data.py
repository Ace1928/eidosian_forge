from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def test_extract_filter_data(self):
    path = uuid.uuid4().hex

    @self.flask_blueprint.route('/%s' % path)
    def return_nothing_interesting():
        return ('OK', 200)
    self._register_blueprint_to_app()
    with self.test_client() as c:
        expected_param = uuid.uuid4().hex
        unexpected_param = uuid.uuid4().hex
        get_path = '/'.join([self.url_prefix, path])
        qs = '%(expected)s=EXPECTED&%(unexpected)s=UNEXPECTED' % {'expected': expected_param, 'unexpected': unexpected_param}
        c.get('%(path)s?%(qs)s' % {'path': get_path, 'qs': qs})
        extracted_filter = self.enforcer._extract_filter_values([expected_param])
        self.assertNotIn(extracted_filter, unexpected_param)
        self.assertIn(expected_param, expected_param)
        self.assertEqual(extracted_filter[expected_param], 'EXPECTED')