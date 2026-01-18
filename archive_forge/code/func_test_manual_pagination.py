import random
import string
from tests.compat import unittest, mock
import boto
def test_manual_pagination(self, num_invals=30, max_items=4):
    """
        Test that paginating manually works properly
        """
    self.assertGreater(num_invals, max_items)
    responses = self._get_mock_responses(num=num_invals, max_items=max_items)
    self.cf.make_request = mock.Mock(side_effect=responses)
    ir = self.cf.get_invalidation_requests('dist-id-here', max_items=max_items)
    all_invals = list(ir)
    self.assertEqual(len(all_invals), max_items)
    while ir.is_truncated:
        ir = self.cf.get_invalidation_requests('dist-id-here', marker=ir.next_marker, max_items=max_items)
        invals = list(ir)
        self.assertLessEqual(len(invals), max_items)
        all_invals.extend(invals)
    remainder = num_invals % max_items
    if remainder != 0:
        self.assertEqual(len(invals), remainder)
    self.assertEqual(len(all_invals), num_invals)