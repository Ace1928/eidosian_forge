import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
def test_call_ok_roles(self):
    dummy_conf = {'auth_uri': 'http://123:5000/v2.0'}
    ec2 = ec2token.EC2Token(app='woot', conf=dummy_conf)
    auth_str = 'Authorization: foo  Credential=foo/bar, SignedHeaders=content-type;host;x-amz-date, Signature=xyz'
    req_env = {'SERVER_NAME': 'heat', 'SERVER_PORT': '8000', 'PATH_INFO': '/v1', 'HTTP_AUTHORIZATION': auth_str}
    dummy_req = self._dummy_GET_request(environ=req_env)
    ok_resp = json.dumps({'token': {'id': 123, 'project': {'name': 'tenant', 'id': 'abcd1234'}, 'roles': [{'name': 'aa'}, {'name': 'bb'}, {'name': 'cc'}]}})
    self._stub_http_connection(headers={'Authorization': auth_str}, response=ok_resp)
    self.assertEqual('woot', ec2.__call__(dummy_req))
    self.assertEqual('aa,bb,cc', dummy_req.headers['X-Roles'])
    requests.post.assert_called_once_with(self.verify_req_url, data=self.verify_data, verify=self.verify_verify, cert=self.verify_cert, headers=self.verify_req_headers)