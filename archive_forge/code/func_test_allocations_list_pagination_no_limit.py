import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def test_allocations_list_pagination_no_limit(self):
    allocations = self.mgr.list(limit=0)
    expect = [('GET', '/v1/allocations', {}, None), ('GET', '/v1/allocations/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(allocations))
    expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/allocations/?limit=1', 'allocations': [ALLOCATION]})
    self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])