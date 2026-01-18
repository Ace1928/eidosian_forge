import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_signed_headers_contains_all_headers_lowercased(self):
    headers = {'Content-Type': 'text/plain', 'Host': 'my_host', 'X-Special-Header': ''}
    signed_headers = self.signer._get_signed_headers(headers)
    self.assertIn('content-type', signed_headers)
    self.assertIn('host', signed_headers)
    self.assertIn('x-special-header', signed_headers)