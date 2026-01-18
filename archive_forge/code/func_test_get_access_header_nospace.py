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
def test_get_access_header_nospace(self):
    req_env = {'HTTP_AUTHORIZATION': 'Authorization: foo  Credential=foo/bar,SignedHeaders=content-type;host;x-amz-date,Signature=xyz'}
    dummy_req = self._dummy_GET_request(environ=req_env)
    ec2 = ec2token.EC2Token(app=None, conf={})
    self.assertEqual('foo', ec2._get_access(dummy_req))