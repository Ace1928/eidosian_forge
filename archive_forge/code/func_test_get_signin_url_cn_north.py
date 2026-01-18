from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_signin_url_cn_north(self):
    self.set_http_response(status_code=200)
    self.service_connection.host = 'iam.cn-north-1.amazonaws.com.cn'
    url = self.service_connection.get_signin_url()
    self.assertEqual(url, 'https://foocorporation.signin.amazonaws.cn/console/ec2')