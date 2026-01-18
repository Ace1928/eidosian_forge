import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GOOGLE, DNS_KEYWORD_PARAMS_GOOGLE
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.dns.drivers.google import GoogleDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_default_scopes(self):
    self.assertIsNone(self.driver.scopes)