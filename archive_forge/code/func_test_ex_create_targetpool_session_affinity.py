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
def test_ex_create_targetpool_session_affinity(self):
    targetpool_name = 'lctargetpool-sticky'
    region = 'us-central1'
    session_affinity = 'CLIENT_IP_PROTO'
    targetpool = self.driver.ex_create_targetpool(targetpool_name, region=region, session_affinity=session_affinity)
    self.assertEqual(targetpool.name, targetpool_name)
    self.assertEqual(targetpool.extra.get('sessionAffinity'), session_affinity)