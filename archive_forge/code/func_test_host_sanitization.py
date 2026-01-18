import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (
def test_host_sanitization(self):
    driver = self.driver_cls(host='example.com')
    self.assertEqual(driver.connection.host, 'example.com')
    driver = self.driver_cls(host='http://example.com')
    self.assertEqual(driver.connection.host, 'example.com')
    driver = self.driver_cls(host='https://example.com')
    self.assertEqual(driver.connection.host, 'example.com')