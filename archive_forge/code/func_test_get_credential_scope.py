import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_credential_scope(self):
    scope = self.signer._get_credential_scope(self.now)
    self.assertEqual(scope, '20150304/my_region/my_service/aws4_request')