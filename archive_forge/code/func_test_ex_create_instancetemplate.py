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
def test_ex_create_instancetemplate(self):
    name = 'my-instance-template1'
    actual = self.driver.ex_create_instancetemplate(name, size='n1-standard-1', image='debian-7', network='default')
    self.assertEqual(actual.name, name)
    self.assertEqual(actual.extra['properties']['machineType'], 'n1-standard-1')