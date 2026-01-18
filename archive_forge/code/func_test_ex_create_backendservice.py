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
def test_ex_create_backendservice(self):
    backendservice_name = 'web-service'
    ig1 = self.driver.ex_get_instancegroup('myinstancegroup', 'us-central1-a')
    backend1 = self.driver.ex_create_backend(ig1)
    ig2 = self.driver.ex_get_instancegroup('myinstancegroup2', 'us-central1-a')
    backend2 = self.driver.ex_create_backend(ig2)
    backendservice = self.driver.ex_create_backendservice(name=backendservice_name, healthchecks=['lchealthcheck'], backends=[backend1, backend2])
    self.assertTrue(isinstance(backendservice, GCEBackendService))
    self.assertEqual(backendservice.name, backendservice_name)
    self.assertEqual(len(backendservice.backends), 2)
    ig_links = [ig1.extra['selfLink'], ig2.extra['selfLink']]
    for be in backendservice.backends:
        self.assertTrue(be['group'] in ig_links)