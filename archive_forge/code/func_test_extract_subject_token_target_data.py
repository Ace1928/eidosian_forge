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
def test_extract_subject_token_target_data(self):
    path = '/v3/auth/tokens'
    body = self._auth_json()
    with self.test_client() as c:
        r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
        token_id = r.headers['X-Subject-Token']
        c.get('/v3', headers={'X-Auth-Token': token_id, 'X-Subject-Token': token_id})
        token = PROVIDER_APIS.token_provider_api.validate_token(token_id)
        subj_token_data = self.enforcer._extract_subject_token_target_data()
        subj_token_data = subj_token_data['token']
        self.assertEqual(token.user_id, subj_token_data['user_id'])
        self.assertIn('user', subj_token_data)
        self.assertIn('domain', subj_token_data['user'])
        self.assertEqual(token.user_domain['id'], subj_token_data['user']['domain']['id'])