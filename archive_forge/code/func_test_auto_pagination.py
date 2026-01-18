import random
import string
from tests.compat import unittest, mock
import boto
def test_auto_pagination(self, num_invals=1024):
    """
        Test that auto-pagination works properly
        """
    max_items = 100
    self.assertGreaterEqual(num_invals, max_items)
    responses = self._get_mock_responses(num=num_invals, max_items=max_items)
    self.cf.make_request = mock.Mock(side_effect=responses)
    ir = self.cf.get_invalidation_requests('dist-id-here')
    self.assertEqual(len(ir._inval_cache), max_items)
    self.assertEqual(len(list(ir)), num_invals)