from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_get_matching_product_for_id(self):
    asins = ['B001UDRNHO', '144930544X']
    response = self.mws.get_matching_product_for_id(MarketplaceId=self.marketplace_id, IdType='ASIN', IdList=asins)
    self.assertEqual(len(response._result), 2)
    for result in response._result:
        self.assertEqual(len(result.Products.Product), 1)