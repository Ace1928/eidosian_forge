import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_request_params_allows_integers_as_value(self):
    self.assertEqual(self.signer._get_request_params({'Action': 'DescribeInstances', 'Port': 22}), 'Action=DescribeInstances&Port=22')