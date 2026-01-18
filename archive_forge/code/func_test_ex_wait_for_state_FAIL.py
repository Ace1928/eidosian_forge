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
def test_ex_wait_for_state_FAIL(self):
    with self.assertRaises(DimensionDataAPIException) as context:
        self.driver.ex_wait_for_state('starting', self.driver.ex_get_node_by_id, id='e75ead52-692f-4314-8725-c8a4f4d13a87', poll_interval=0.1, timeout=0.1)
    self.assertEqual(context.exception.code, 'running')
    self.assertTrue('timed out' in context.exception.msg)