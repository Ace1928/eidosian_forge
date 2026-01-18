import sys
from libcloud.test import unittest
from libcloud.test.compute.test_cloudstack import CloudStackCommonTestCase
from libcloud.compute.drivers.auroracompute import AuroraComputeRegion, AuroraComputeNodeDriver
def test_api_host(self):
    driver = self.driver_klass('invalid', 'invalid')
    self.assertEqual(driver.host, 'api.auroracompute.eu')