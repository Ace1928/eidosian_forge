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
def test_ex_create_targetinstance(self):
    targetinstance_name = 'lctargetinstance'
    zone = 'us-central1-a'
    node = self.driver.ex_get_node('node-name', zone)
    targetinstance = self.driver.ex_create_targetinstance(targetinstance_name, zone=zone, node=node)
    self.assertEqual(targetinstance.name, targetinstance_name)
    self.assertEqual(targetinstance.zone.name, zone)