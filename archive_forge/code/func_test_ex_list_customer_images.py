import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_ex_list_customer_images(self):
    images = self.driver.ex_list_customer_images()
    self.assertEqual(len(images), 3)
    self.assertEqual(images[0].name, 'ImportedCustomerImage')
    self.assertEqual(images[0].id, '5234e5c7-01de-4411-8b6e-baeb8d91cf5d')
    self.assertEqual(images[0].extra['location'].id, 'NA9')
    self.assertEqual(images[0].extra['cpu'].cpu_count, 4)
    self.assertEqual(images[0].extra['OS_displayName'], 'REDHAT6/64')