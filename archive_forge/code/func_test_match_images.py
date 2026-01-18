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
def test_match_images(self):
    project = 'debian-cloud'
    image = self.driver._match_images(project, 'debian-7')
    self.assertEqual(image.name, 'debian-7-wheezy-v20131120')
    image = self.driver._match_images(project, 'backports')
    self.assertEqual(image.name, 'backports-debian-7-wheezy-v20131127')