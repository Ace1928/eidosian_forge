from __future__ import print_function
import sys
import os
import os.path
from datetime import datetime, timedelta
from boto.mws.connection import MWSConnection
from tests.compat import unittest
@unittest.skipUnless(simple and isolator, 'skipping simple test')
def test_list_inventory_supply(self):
    asof = (datetime.today() - timedelta(days=30)).isoformat()
    response = self.mws.list_inventory_supply(QueryStartDateTime=asof, ResponseGroup='Basic')
    self.assertTrue(hasattr(response._result, 'InventorySupplyList'))