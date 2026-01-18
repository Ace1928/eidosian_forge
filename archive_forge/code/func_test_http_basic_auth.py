import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (
def test_http_basic_auth(self):
    driver = self.driver_cls(key='username', secret='password')
    self.assertEqual(driver.connectionCls, KubernetesBasicAuthConnection)
    self.assertEqual(driver.connection.user_id, 'username')
    self.assertEqual(driver.connection.key, 'password')
    auth_string = base64.b64encode(b('{}:{}'.format('username', 'password'))).decode('utf-8')
    headers = driver.connection.add_default_headers({})
    self.assertEqual(headers['Content-Type'], 'application/json')
    self.assertEqual(headers['Authorization'], 'Basic %s' % auth_string)