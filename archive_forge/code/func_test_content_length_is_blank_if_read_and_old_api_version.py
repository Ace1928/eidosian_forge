import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.azure import AzureConnection
def test_content_length_is_blank_if_read_and_old_api_version(self):
    headers = {}
    method = 'GET'
    self.conn.API_VERSION = '2011-08-18'
    values = self.conn._format_special_header_values(headers, method)
    self.assertEqual(values[2], '')