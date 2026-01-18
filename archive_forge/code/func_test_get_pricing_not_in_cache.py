import sys
import os.path
import unittest
import libcloud.pricing
def test_get_pricing_not_in_cache(self):
    try:
        libcloud.pricing.get_pricing(driver_type='compute', driver_name='inexistent', pricing_file_path=PRICING_FILE_PATH)
    except KeyError:
        pass
    else:
        self.fail('Invalid driver provided, but an exception was not thrown')