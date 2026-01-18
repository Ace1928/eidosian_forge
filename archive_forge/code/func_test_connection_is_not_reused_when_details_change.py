import sys
import unittest
from unittest.mock import Mock, patch
from libcloud.common.base import LibcloudConnection
from libcloud.common.openstack import OpenStackBaseConnection
@patch('libcloud.test.common.test_openstack.OpenStackBaseConnection.connect', Mock())
def test_connection_is_not_reused_when_details_change(self):
    url = 'https://example.com'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 1)
    url = 'https://example.com'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 1)
    url = 'https://example.com:80'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 2)
    url = 'http://example.com:80'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 3)
    url = 'http://exxample.com:80'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 4)
    url = 'http://exxample.com:81'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 5)
    url = 'http://exxample.com:81'
    self.connection._set_up_connection_info(url=url)
    self.assertEqual(self.connection.connect.call_count, 5)