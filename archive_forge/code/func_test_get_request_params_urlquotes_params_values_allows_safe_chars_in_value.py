import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_request_params_urlquotes_params_values_allows_safe_chars_in_value(self):
    self.assertEqual('Action=a~b.c_d-e', self.signer._get_request_params({'Action': 'a~b.c_d-e'}))