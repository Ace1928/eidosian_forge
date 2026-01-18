import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_log_req(self):
    self.logger = self.useFixture(fixtures.FakeLogger(format='%(message)s', level=logging.DEBUG, nuke_handlers=True))
    kwargs = {'headers': {'X-Foo': 'bar'}, 'data': '{"auth": {"tenantName": "fakeService", "passwordCredentials": {"username": "fakeUser", "password": "fakePassword"}}}'}
    cs = cinderclient.client.HTTPClient('user', None, None, 'http://127.0.0.1:5000')
    cs.http_log_debug = True
    cs.http_log_req('PUT', kwargs)
    output = self.logger.output.split('\n')
    self.assertNotIn('fakePassword', output[1])
    self.assertIn('fakeUser', output[1])