import sys
import os.path
import unittest
import libcloud.pricing
def test_get_gce_image_price_SLES_SAP_image(self):
    cores = 2
    image_name = 'sles-15-sap-v20220126'
    size_name = 'n2d-highcpu-2'
    prices = libcloud.pricing.get_pricing('compute', 'gce_images')
    correct_price = float(prices['SLES for SAP']['1-2vcpu']['price'])
    fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
    self.assertTrue(fetched_price == correct_price)