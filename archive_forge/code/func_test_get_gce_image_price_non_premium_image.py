import sys
import os.path
import unittest
import libcloud.pricing
def test_get_gce_image_price_non_premium_image(self):
    image_name = 'debian-10-buster-v20220519'
    cores = 4
    size_name = 'c2d-standard-4'
    price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
    self.assertTrue(price == 0)