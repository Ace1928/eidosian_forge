import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def test_allocations_show(self):
    allocation = self.mgr.get(ALLOCATION['uuid'])
    expect = [('GET', '/v1/allocations/%s' % ALLOCATION['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(ALLOCATION['uuid'], allocation.uuid)
    self.assertEqual(ALLOCATION['name'], allocation.name)
    self.assertEqual(ALLOCATION['owner'], allocation.owner)
    self.assertEqual(ALLOCATION['node_uuid'], allocation.node_uuid)
    self.assertEqual(ALLOCATION['state'], allocation.state)
    self.assertEqual(ALLOCATION['resource_class'], allocation.resource_class)
    expected_resp = ({}, ALLOCATION)
    self.assertEqual(expected_resp, self.api.responses['/v1/allocations/%s' % ALLOCATION['uuid']]['GET'])