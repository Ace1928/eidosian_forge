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
def test_call_ok_auth_uri_ec2authtoken(self):
    dummy_url = 'http://123:5000/v2.0'
    cfg.CONF.set_default('auth_uri', dummy_url, group='ec2authtoken')
    ec2 = ec2token.EC2Token(app='woot', conf={})
    params = {'AWSAccessKeyId': 'foo', 'Signature': 'xyz'}
    req_env = {'SERVER_NAME': 'heat', 'SERVER_PORT': '8000', 'PATH_INFO': '/v1'}
    dummy_req = self._dummy_GET_request(params, req_env)
    ok_resp = json.dumps({'token': {'project': {'name': 'tenant', 'id': 'abcd1234'}}})
    self._stub_http_connection(response=ok_resp, params={'AWSAccessKeyId': 'foo'})
    self.assertEqual('woot', ec2.__call__(dummy_req))
    requests.post.assert_called_with(self.verify_req_url, data=self.verify_data, verify=self.verify_verify, cert=self.verify_cert, headers=self.verify_req_headers)