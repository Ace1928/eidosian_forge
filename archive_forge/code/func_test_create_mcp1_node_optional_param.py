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
def test_create_mcp1_node_optional_param(self):
    root_pw = NodeAuthPassword('pass123')
    image = self.driver.list_images()[0]
    network = self.driver.ex_list_networks()[0]
    cpu_spec = DimensionDataServerCpuSpecification(cpu_count='4', cores_per_socket='2', performance='STANDARD')
    disks = [DimensionDataServerDisk(scsi_id='0', speed='HIGHPERFORMANCE')]
    node = self.driver.create_node(name='test2', image=image, auth=root_pw, ex_description='test2 node', ex_network=network, ex_is_started=False, ex_memory_gb=8, ex_disks=disks, ex_cpu_specification=cpu_spec, ex_primary_dns='10.0.0.5', ex_secondary_dns='10.0.0.6')
    self.assertEqual(node.id, 'e75ead52-692f-4314-8725-c8a4f4d13a87')
    self.assertEqual(node.extra['status'].action, 'DEPLOY_SERVER')