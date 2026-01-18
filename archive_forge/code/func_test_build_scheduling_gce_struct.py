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
def test_build_scheduling_gce_struct(self):
    self.assertFalse(self.driver._build_scheduling_gce_struct(None, None, None))
    self.assertRaises(ValueError, self.driver._build_service_account_gce_struct, 'on_host_maintenance="foobar"')
    self.assertRaises(ValueError, self.driver._build_service_account_gce_struct, 'on_host_maintenance="MIGRATE"', 'preemptible=True')
    self.assertRaises(ValueError, self.driver._build_service_account_gce_struct, 'automatic_restart="True"', 'preemptible=True')
    actual = self.driver._build_scheduling_gce_struct('TERMINATE', True, False)
    self.assertTrue('automaticRestart' in actual and actual['automaticRestart'] is True)
    self.assertTrue('onHostMaintenance' in actual and actual['onHostMaintenance'] == 'TERMINATE')
    self.assertTrue('preemptible' in actual)
    self.assertFalse(actual['preemptible'])