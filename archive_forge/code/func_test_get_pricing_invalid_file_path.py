import sys
import os.path
import unittest
import libcloud.pricing
def test_get_pricing_invalid_file_path(self):
    try:
        libcloud.pricing.get_pricing(driver_type='compute', driver_name='bar', pricing_file_path='inexistent.json')
    except OSError:
        pass
    else:
        self.fail('Invalid pricing file path provided, but an exception was not thrown')