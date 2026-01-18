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
def assertValidCredential(self, entity, ref=None):
    self.assertIsNotNone(entity.get('user_id'))
    self.assertIsNotNone(entity.get('blob'))
    self.assertIsNotNone(entity.get('type'))
    self.assertNotIn('key_hash', entity)
    self.assertNotIn('encrypted_blob', entity)
    if ref:
        self.assertEqual(ref['user_id'], entity['user_id'])
        self.assertEqual(ref['blob'], entity['blob'])
        self.assertEqual(ref['type'], entity['type'])
        self.assertEqual(ref.get('project_id'), entity.get('project_id'))
    return entity