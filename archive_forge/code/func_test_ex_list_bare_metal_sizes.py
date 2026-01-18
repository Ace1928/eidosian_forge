import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_ex_list_bare_metal_sizes(self):
    sizes = self.driver.ex_list_bare_metal_sizes()
    self.assertEqual(len(sizes), 4)
    for size in sizes:
        self.assertIsInstance(size.extra['cpu_count'], int)
        self.assertIsInstance(size.extra['cpu_threads'], int)
        self.assertIsInstance(size.extra['cpu_model'], str)