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
def assertValidCatalogResponse(self, resp, *args, **kwargs):
    self.assertEqual(set(['catalog', 'links']), set(resp.json.keys()))
    self.assertValidCatalog(resp.json['catalog'])
    self.assertIn('links', resp.json)
    self.assertIsInstance(resp.json['links'], dict)
    self.assertEqual(['self'], list(resp.json['links'].keys()))
    self.assertEqual('http://localhost/v3/auth/catalog', resp.json['links']['self'])