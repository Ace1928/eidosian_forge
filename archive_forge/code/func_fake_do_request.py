import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def fake_do_request(cls, url, method, headers=None, body=None):
    if not url.rstrip('/').endswith('v2.0/tokens') or url.count('2.0') != 1:
        self.fail('Invalid v2.0 token path (%s)' % url)
    creds = jsonutils.loads(body)['auth']
    username = creds['passwordCredentials']['username']
    password = creds['passwordCredentials']['password']
    tenant = creds['tenantName']
    resp = webob.Response()
    if username != 'user1' or password != 'pass' or tenant != 'tenant-ok':
        resp.status = http.UNAUTHORIZED
    else:
        resp.status = http.OK
        body = mock_token.token
    return (FakeResponse(resp), jsonutils.dumps(body))