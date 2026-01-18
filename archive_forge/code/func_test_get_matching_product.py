from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_get_matching_product(self):
    asin = 'B001UDRNHO'
    response = self.mws.get_matching_product(MarketplaceId=self.marketplace_id, ASINList=[asin])
    attributes = response._result[0].Product.AttributeSets.ItemAttributes
    self.assertEqual(attributes[0].Label, 'Serengeti')