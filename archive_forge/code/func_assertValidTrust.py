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
def assertValidTrust(self, entity, ref=None, summary=False):
    self.assertIsNotNone(entity.get('trustor_user_id'))
    self.assertIsNotNone(entity.get('trustee_user_id'))
    self.assertIsNotNone(entity.get('impersonation'))
    self.assertIn('expires_at', entity)
    if entity['expires_at'] is not None:
        self.assertValidISO8601ExtendedFormatDatetime(entity['expires_at'])
    if summary:
        self.assertNotIn('roles', entity)
        self.assertIn('project_id', entity)
    else:
        for role in entity['roles']:
            self.assertIsNotNone(role)
            self.assertValidEntity(role, keys_to_check=['name'])
            self.assertValidRole(role)
        self.assertValidListLinks(entity.get('roles_links'))
        self.assertIn('v3/OS-TRUST/trusts', entity.get('links')['self'])
        has_roles = bool(entity.get('roles'))
        has_project = bool(entity.get('project_id'))
        self.assertFalse(has_roles ^ has_project)
    if ref:
        self.assertEqual(ref['trustor_user_id'], entity['trustor_user_id'])
        self.assertEqual(ref['trustee_user_id'], entity['trustee_user_id'])
        self.assertEqual(ref['project_id'], entity['project_id'])
        if entity.get('expires_at') or ref.get('expires_at'):
            entity_exp = self.assertValidISO8601ExtendedFormatDatetime(entity['expires_at'])
            ref_exp = self.assertValidISO8601ExtendedFormatDatetime(ref['expires_at'])
            self.assertCloseEnoughForGovernmentWork(entity_exp, ref_exp)
        else:
            self.assertEqual(ref.get('expires_at'), entity.get('expires_at'))
    return entity