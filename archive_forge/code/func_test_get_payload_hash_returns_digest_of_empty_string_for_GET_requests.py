import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_payload_hash_returns_digest_of_empty_string_for_GET_requests(self):
    SignedAWSConnection.method = 'GET'
    self.assertEqual(self.signer._get_payload_hash(method='GET'), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')