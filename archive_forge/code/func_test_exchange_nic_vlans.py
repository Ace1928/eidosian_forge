import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_exchange_nic_vlans(self):
    success = self.driver.ex_exchange_nic_vlans(nic_id_1='a4b4b42b-ccb5-416f-b052-ce7cb7fdff12', nic_id_2='b39d09b8-ea65-424a-8fa6-c6f5a98afc69')
    self.assertTrue(success)