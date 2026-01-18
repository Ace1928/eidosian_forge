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
def assertValidListResponse(self, resp, key, entity_validator, ref=None, expected_length=None, keys_to_check=None, resource_url=None):
    """Make assertions common to all API list responses.

        If a reference is provided, it's ID will be searched for in the
        response, and asserted to be equal.

        """
    entities = resp.result.get(key)
    self.assertIsNotNone(entities)
    if expected_length is not None:
        self.assertEqual(expected_length, len(entities))
    elif ref is not None:
        self.assertNotEmpty(entities)
    self.assertValidListLinks(resp.result.get('links'), resource_url=resource_url)
    for entity in entities:
        self.assertIsNotNone(entity)
        self.assertValidEntity(entity, keys_to_check=keys_to_check)
        entity_validator(entity)
    if ref:
        entity = [x for x in entities if x['id'] == ref['id']][0]
        self.assertValidEntity(entity, ref=ref, keys_to_check=keys_to_check)
        entity_validator(entity, ref)
    return entities