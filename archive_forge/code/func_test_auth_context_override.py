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
def test_auth_context_override(self):
    overridden_context = 'OVERRIDDEN_CONTEXT'
    token = uuid.uuid4().hex
    extra_environ = {authorization.AUTH_CONTEXT_ENV: overridden_context}
    req = self._middleware_request(token, extra_environ=extra_environ)
    self.assertEqual(overridden_context, req.environ.get(authorization.AUTH_CONTEXT_ENV))