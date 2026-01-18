import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_create_sslcertificate(self):
    ssl_name = 'example'
    private_key = '-----BEGIN RSA PRIVATE KEY-----\nfoobar==\n-----END RSA PRIVATE KEY-----\n'
    certificate = '-----BEGIN CERTIFICATE-----\nfoobar==\n-----END CERTIFICATE-----\n'
    ssl = self.driver.ex_create_sslcertificate(ssl_name, certificate=certificate, private_key=private_key)
    self.assertEqual(ssl_name, ssl.name)
    self.assertEqual(certificate, ssl.certificate)