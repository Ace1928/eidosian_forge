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
def test_ex_set_usage_export_bucket(self):
    self.assertRaises(ValueError, self.driver.ex_set_usage_export_bucket, 'foo')
    bucket_name = 'gs://foo'
    self.driver.ex_set_usage_export_bucket(bucket_name)
    bucket_name = 'https://www.googleapis.com/foo'
    self.driver.ex_set_usage_export_bucket(bucket_name)
    project = GCEProject(id=None, name=None, metadata=None, quotas=None, driver=self.driver)
    project.set_usage_export_bucket(bucket=bucket_name)