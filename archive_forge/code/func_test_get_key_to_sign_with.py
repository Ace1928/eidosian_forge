import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_key_to_sign_with(self):

    def _sign(key, msg, hex=False):
        return '{}|{}'.format(key, msg)
    with mock.patch('libcloud.common.aws._sign', new=_sign):
        key = self.signer._get_key_to_sign_with(self.now)
    self.assertEqual(key, 'AWS4my_secret|20150304|my_region|my_service|aws4_request')