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
def test_create_mcp1_node_no_network(self):
    rootPw = NodeAuthPassword('pass123')
    image = self.driver.list_images()[0]
    with self.assertRaises(InvalidRequestError):
        self.driver.create_node(name='test2', image=image, auth=rootPw, ex_description='test2 node', ex_network=None, ex_is_started=False)