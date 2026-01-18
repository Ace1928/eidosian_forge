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
def test_list_nodes_response_strings_ALLFILTERS(self):
    DimensionDataMockHttp.type = 'ALLFILTERS'
    ret = self.driver.list_nodes(ex_location='fake_loc', ex_name='fake_name', ex_ipv6='fake_ipv6', ex_ipv4='fake_ipv4', ex_vlan='fake_vlan', ex_image='fake_image', ex_deployed=True, ex_started=True, ex_state='fake_state', ex_network='fake_network', ex_network_domain='fake_network_domain')
    self.assertTrue(isinstance(ret, list))
    self.assertEqual(len(ret), 7)
    node = ret[3]
    self.assertTrue(isinstance(node.extra['disks'], list))
    self.assertTrue(isinstance(node.extra['disks'][0], DimensionDataServerDisk))
    self.assertEqual(node.size.id, '1')
    self.assertEqual(node.image.id, '3ebf3c0f-90fe-4a8b-8585-6e65b316592c')
    self.assertEqual(node.image.name, 'WIN2008S/32')
    disk = node.extra['disks'][0]
    self.assertEqual(disk.id, 'c2e1f199-116e-4dbc-9960-68720b832b0a')
    self.assertEqual(disk.scsi_id, 0)
    self.assertEqual(disk.size_gb, 50)
    self.assertEqual(disk.speed, 'STANDARD')
    self.assertEqual(disk.state, 'NORMAL')