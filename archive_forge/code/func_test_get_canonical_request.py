import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_request(self):
    req = self.signer._get_canonical_request({'Action': 'DescribeInstances', 'Version': '2013-10-15'}, {'Accept-Encoding': 'gzip,deflate', 'User-Agent': 'My-UA'}, method='GET', path='/my_action/', data=None)
    self.assertEqual(req, 'GET\n/my_action/\nAction=DescribeInstances&Version=2013-10-15\naccept-encoding:gzip,deflate\nuser-agent:My-UA\n\naccept-encoding;user-agent\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')