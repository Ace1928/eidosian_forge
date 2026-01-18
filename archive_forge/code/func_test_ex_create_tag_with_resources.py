import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_create_tag_with_resources(self):
    CloudSigmaMockHttp.type = 'WITH_RESOURCES'
    resource_uuids = ['1']
    tag = self.driver.ex_create_tag(name='test tag 3', resource_uuids=resource_uuids)
    self.assertEqual(tag.name, 'test tag 3')
    self.assertEqual(tag.resources, resource_uuids)