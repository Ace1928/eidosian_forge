import sys
import os.path
import unittest
import libcloud.pricing
def test_get_pricing_invalid_driver_type(self):
    try:
        libcloud.pricing.get_pricing(driver_type='invalid_type', driver_name='bar', pricing_file_path='inexistent.json')
    except AttributeError:
        pass
    else:
        self.fail('Invalid driver_type provided, but an exception was not thrown')