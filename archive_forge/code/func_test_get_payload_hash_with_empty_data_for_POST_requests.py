import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_payload_hash_with_empty_data_for_POST_requests(self):
    SignedAWSConnection.method = 'POST'
    self.assertEqual(self.signer._get_payload_hash(method='POST'), UNSIGNED_PAYLOAD)