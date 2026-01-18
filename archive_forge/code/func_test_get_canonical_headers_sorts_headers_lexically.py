import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_headers_sorts_headers_lexically(self):
    headers = {'accept-encoding': 'gzip,deflate', 'host': 'my_host', '1st-header': '2', 'x-amz-date': '20150304T173452Z', 'user-agent': 'my-ua'}
    self.assertEqual(self.signer._get_canonical_headers(headers), '1st-header:2\naccept-encoding:gzip,deflate\nhost:my_host\nuser-agent:my-ua\nx-amz-date:20150304T173452Z\n')