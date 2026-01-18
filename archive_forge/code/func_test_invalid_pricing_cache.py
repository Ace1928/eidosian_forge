import sys
import os.path
import unittest
import libcloud.pricing
def test_invalid_pricing_cache(self):
    libcloud.pricing.PRICING_DATA['compute']['foo'] = {2: 2}
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
    libcloud.pricing.invalidate_pricing_cache()
    self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])