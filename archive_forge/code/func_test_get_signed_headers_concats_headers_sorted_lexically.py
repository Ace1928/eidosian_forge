import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_signed_headers_concats_headers_sorted_lexically(self):
    headers = {'Host': 'my_host', 'X-Special-Header': '', '1St-Header': '2', 'Content-Type': 'text/plain'}
    signed_headers = self.signer._get_signed_headers(headers)
    self.assertEqual(signed_headers, '1st-header;content-type;host;x-special-header')