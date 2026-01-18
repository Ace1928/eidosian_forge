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
def test_ex_create_node_uncustomized_mcp2_using_ipv4(self):
    node = self.driver.ex_create_node_uncustomized(name='test_server_05', image='fake_customer_image', ex_network_domain='fakenetworkdomain', ex_is_started=False, ex_description=None, ex_cluster_id=None, ex_cpu_specification=None, ex_memory_gb=None, ex_primary_nic_private_ipv4='10.0.0.1', ex_primary_nic_vlan=None, ex_primary_nic_network_adapter=None, ex_additional_nics=None, ex_disks=None, ex_tagid_value_pairs=None, ex_tagname_value_pairs=None)
    self.assertEqual(node.id, 'e75ead52-692f-4314-8725-c8a4f4d13a87')