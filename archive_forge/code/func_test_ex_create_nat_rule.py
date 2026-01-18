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
def test_ex_create_nat_rule(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rule = self.driver.ex_create_nat_rule(net, '1.2.3.4', '4.3.2.1')
    self.assertEqual(rule.id, 'd31c2db0-be6b-4d50-8744-9a7a534b5fba')