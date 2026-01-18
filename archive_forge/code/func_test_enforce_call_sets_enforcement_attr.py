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
def test_enforce_call_sets_enforcement_attr(self):
    token_path = '/v3/auth/tokens'
    auth_json = self._auth_json()
    with self.test_client() as c:
        r = c.post(token_path, json=auth_json, expected_status_code=201)
        token_id = r.headers.get('X-Subject-Token')
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={'X-Auth-Token': token_id})
        self.assertFalse(hasattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR))
        setattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR, False)
        self.enforcer.enforce_call(action='example:allowed')
        self.assertEqual(getattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR), True)
        setattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR, False)
        self.assertRaises(exception.ForbiddenAction, self.enforcer.enforce_call, action='example:denied')
        self.assertEqual(getattr(flask.g, rbac_enforcer.enforcer._ENFORCEMENT_CHECK_ATTR), True)