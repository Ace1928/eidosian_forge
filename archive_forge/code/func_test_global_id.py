import copy
from unittest import mock
from keystoneauth1 import session
from oslo_utils import uuidutils
import novaclient.api_versions
import novaclient.client
import novaclient.extension
from novaclient.tests.unit import utils
import novaclient.v2.client
def test_global_id(self):
    global_id = 'req-%s' % uuidutils.generate_uuid()
    self.requests_mock.get('http://no.where')
    client = novaclient.client.SessionClient(session=session.Session(), global_request_id=global_id)
    client.request('http://no.where', 'GET')
    headers = self.requests_mock.last_request.headers
    self.assertEqual(headers['X-OpenStack-Request-ID'], global_id)