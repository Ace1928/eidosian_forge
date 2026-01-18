import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def test_accept_type_handling(self):

    def make_request(accept_types=None):
        client = TestClient(self.public_app)
        headers = None
        if accept_types:
            headers = {'Accept': accept_types}
        resp = client.get('/v3', headers=headers)
        self.assertThat(resp.status, tt_matchers.Equals('200 OK'))
        return resp.headers['Content-Type']
    JSON = discovery.MimeTypes.JSON
    JSON_HOME = discovery.MimeTypes.JSON_HOME
    JSON_MATCHER = tt_matchers.Equals(JSON)
    JSON_HOME_MATCHER = tt_matchers.Equals(JSON_HOME)
    self.assertThat(make_request(), JSON_MATCHER)
    self.assertThat(make_request(JSON), JSON_MATCHER)
    self.assertThat(make_request(JSON_HOME), JSON_HOME_MATCHER)
    accept_types = '%s, %s' % (JSON, JSON_HOME)
    self.assertThat(make_request(accept_types), JSON_MATCHER)
    accept_types = '%s, %s' % (JSON_HOME, JSON)
    self.assertThat(make_request(accept_types), JSON_MATCHER)
    accept_types = '%s, %s;q=0.5' % (JSON_HOME, JSON)
    self.assertThat(make_request(accept_types), JSON_HOME_MATCHER)
    self.assertThat(make_request(self.getUniqueString()), JSON_MATCHER)