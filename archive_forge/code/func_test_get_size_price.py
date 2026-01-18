import sys
import os.path
import unittest
import libcloud.pricing
def test_get_size_price(self):
    libcloud.pricing.PRICING_DATA['compute']['foo'] = {2: 2, '3': 3}
    price1 = libcloud.pricing.get_size_price(driver_type='compute', driver_name='foo', size_id=2)
    price2 = libcloud.pricing.get_size_price(driver_type='compute', driver_name='foo', size_id='3')
    self.assertEqual(price1, 2)
    self.assertEqual(price2, 3)