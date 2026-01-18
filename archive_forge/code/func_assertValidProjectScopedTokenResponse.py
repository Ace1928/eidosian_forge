import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def assertValidProjectScopedTokenResponse(self, r, *args, **kwargs):
    token = self.assertValidScopedTokenResponse(r, *args, **kwargs)
    project_scoped_token_schema = self.generate_token_schema(project_scoped=True)
    if token.get('OS-TRUST:trust'):
        trust_properties = {'OS-TRUST:trust': {'type': ['object'], 'required': ['id', 'impersonation', 'trustor_user', 'trustee_user'], 'properties': {'id': {'type': 'string'}, 'impersonation': {'type': 'boolean'}, 'trustor_user': {'type': 'object', 'required': ['id'], 'properties': {'id': {'type': 'string'}}, 'additionalProperties': False}, 'trustee_user': {'type': 'object', 'required': ['id'], 'properties': {'id': {'type': 'string'}}, 'additionalProperties': False}}, 'additionalProperties': False}}
        project_scoped_token_schema['properties'].update(trust_properties)
    validator_object = validators.SchemaValidator(project_scoped_token_schema)
    validator_object.validate(token)
    self.assertEqual(self.role_id, token['roles'][0]['id'])
    return token