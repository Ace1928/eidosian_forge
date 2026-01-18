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
def test_get_json_home(self):
    resp = self.get('/', convert=False, headers={'Accept': 'application/json-home'})
    self.assertThat(resp.headers['Content-Type'], matchers.Equals('application/json-home'))
    resp_data = jsonutils.loads(resp.body)
    for rel in self.JSON_HOME_DATA:
        self.assertThat(resp_data['resources'][rel], matchers.Equals(self.JSON_HOME_DATA[rel]))