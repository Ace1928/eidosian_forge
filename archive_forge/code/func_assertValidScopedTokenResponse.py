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
def assertValidScopedTokenResponse(self, r, *args, **kwargs):
    require_catalog = kwargs.pop('require_catalog', True)
    endpoint_filter = kwargs.pop('endpoint_filter', False)
    ep_filter_assoc = kwargs.pop('ep_filter_assoc', 0)
    is_admin_project = kwargs.pop('is_admin_project', None)
    token = self.assertValidTokenResponse(r, *args, **kwargs)
    if require_catalog:
        endpoint_num = 0
        self.assertIn('catalog', token)
        if isinstance(token['catalog'], list):
            for service in token['catalog']:
                for endpoint in service['endpoints']:
                    self.assertNotIn('enabled', endpoint)
                    self.assertNotIn('legacy_endpoint_id', endpoint)
                    self.assertNotIn('service_id', endpoint)
                    endpoint_num += 1
        if endpoint_filter:
            self.assertEqual(ep_filter_assoc, endpoint_num)
    else:
        self.assertNotIn('catalog', token)
    self.assertIn('roles', token)
    self.assertTrue(token['roles'])
    for role in token['roles']:
        self.assertIn('id', role)
        self.assertIn('name', role)
    self.assertIs(is_admin_project, token.get('is_admin_project'))
    return token