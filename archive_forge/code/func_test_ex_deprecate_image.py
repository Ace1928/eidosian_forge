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
def test_ex_deprecate_image(self):
    dep_ts = '2064-03-11T20:18:36.194-07:00'
    obs_ts = '2074-03-11T20:18:36.194-07:00'
    del_ts = '2084-03-11T20:18:36.194-07:00'
    image = self.driver.ex_get_image('debian-7-wheezy-v20131014')
    deprecated = image.deprecate('debian-7', 'DEPRECATED', deprecated=dep_ts, obsolete=obs_ts, deleted=del_ts)
    self.assertTrue(deprecated)
    self.assertEqual(image.extra['deprecated']['deprecated'], dep_ts)
    self.assertEqual(image.extra['deprecated']['obsolete'], obs_ts)
    self.assertEqual(image.extra['deprecated']['deleted'], del_ts)