import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.azure import AzureConnection
def test_content_length_is_used_if_set(self):
    headers = {'content-length': '123'}
    method = 'PUT'
    values = self.conn._format_special_header_values(headers, method)
    self.assertEqual(values[2], '123')