from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_get_product_categories_for_asin(self):
    asin = '144930544X'
    response = self.mws.get_product_categories_for_asin(MarketplaceId=self.marketplace_id, ASIN=asin)
    self.assertEqual(len(response._result.Self), 3)
    categoryids = [x.ProductCategoryId for x in response._result.Self]
    self.assertSequenceEqual(categoryids, ['285856', '21', '491314'])