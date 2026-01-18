import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_request_params_urlquotes_params_values(self):
    self.assertEqual(self.signer._get_request_params({'Action': 'DescribeInstances&Addresses', 'Port-Range': '2000 3000'}), 'Action=DescribeInstances%26Addresses&Port-Range=2000%203000')