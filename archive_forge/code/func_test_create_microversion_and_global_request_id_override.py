import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_create_microversion_and_global_request_id_override(self):
    resource = self.manager.create(**CREATE_TESTABLE_RESOURCE, os_ironic_api_version='1.22', global_request_id=REQ_ID)
    expect = [('POST', '/v1/testableresources', {'X-OpenStack-Ironic-API-Version': '1.22', 'X-Openstack-Request-Id': REQ_ID}, CREATE_TESTABLE_RESOURCE)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(resource)
    self.assertIsInstance(resource, TestableResource)