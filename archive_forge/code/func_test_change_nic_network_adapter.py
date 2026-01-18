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
def test_change_nic_network_adapter(self):
    success = self.driver.ex_change_nic_network_adapter(nic_id='0c55c269-20a5-4fec-8054-22a245a48fe4', network_adapter_name='E1000')
    self.assertTrue(success)