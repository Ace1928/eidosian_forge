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
def assertValidListLinks(self, links, resource_url=None):
    self.assertIsNotNone(links)
    self.assertIsNotNone(links.get('self'))
    self.assertThat(links['self'], matchers.StartsWith('http://localhost'))
    if resource_url:
        self.assertThat(links['self'], matchers.EndsWith(resource_url))
    self.assertIn('next', links)
    if links['next'] is not None:
        self.assertThat(links['next'], matchers.StartsWith('http://localhost'))
    self.assertIn('previous', links)
    if links['previous'] is not None:
        self.assertThat(links['previous'], matchers.StartsWith('http://localhost'))