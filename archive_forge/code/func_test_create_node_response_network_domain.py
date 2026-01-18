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
def test_create_node_response_network_domain(self):
    rootPw = NodeAuthPassword('pass123')
    location = self.driver.ex_get_location_by_id('NA9')
    image = self.driver.list_images(location=location)[0]
    network_domain = self.driver.ex_list_network_domains(location=location)[0]
    vlan = self.driver.ex_list_vlans(location=location)[0]
    cpu = DimensionDataServerCpuSpecification(cpu_count=4, cores_per_socket=1, performance='HIGHPERFORMANCE')
    node = self.driver.create_node(name='test2', image=image, auth=rootPw, ex_description='test2 node', ex_network_domain=network_domain, ex_vlan=vlan, ex_is_started=False, ex_cpu_specification=cpu, ex_memory_gb=4)
    self.assertEqual(node.id, 'e75ead52-692f-4314-8725-c8a4f4d13a87')
    self.assertEqual(node.extra['status'].action, 'DEPLOY_SERVER')