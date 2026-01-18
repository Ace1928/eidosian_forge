from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_list_matching_products(self):
    response = self.mws.list_matching_products(MarketplaceId=self.marketplace_id, Query='boto')
    products = response._result.Products
    self.assertTrue(len(products))