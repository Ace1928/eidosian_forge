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
def test_call_auth_nosig(self):
    req_env = {'HTTP_AUTHORIZATION': 'Authorization: foo  Credential=foo/bar, SignedHeaders=content-type;host;x-amz-date'}
    dummy_req = self._dummy_GET_request(environ=req_env)
    ec2 = ec2token.EC2Token(app='xyz', conf={})
    self.assertRaises(exception.HeatIncompleteSignatureError, ec2.__call__, dummy_req)