import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def test_allocations_list_by_owner(self):
    allocations = self.mgr.list(owner=ALLOCATION2['owner'])
    expect = [('GET', '/v1/allocations/?owner=%s' % ALLOCATION2['owner'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(allocations))
    expected_resp = ({}, {'allocations': [ALLOCATION, ALLOCATION2]})
    self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])