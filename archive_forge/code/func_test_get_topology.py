from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_get_topology(self):
    sot = server.Server(**EXAMPLE)

    class FakeEndpointData:
        min_microversion = '2.1'
        max_microversion = '2.78'
    self.sess.get_endpoint_data.return_value = FakeEndpointData()
    self.sess.default_microversion = None
    response = mock.Mock()
    topology = {'nodes': [{'cpu_pinning': {'0': 0, '1': 5}, 'host_node': 0, 'memory_mb': 1024, 'siblings': [[0, 1]], 'vcpu_set': [0, 1]}, {'cpu_pinning': {'2': 1, '3': 8}, 'host_node': 1, 'memory_mb': 2048, 'siblings': [[2, 3]], 'vcpu_set': [2, 3]}], 'pagesize_kb': 4}
    response.status_code = 200
    response.json.return_value = topology
    self.sess.get.return_value = response
    fetched_topology = sot.fetch_topology(self.sess)
    url = 'servers/IDENTIFIER/topology'
    self.sess.get.assert_called_with(url)
    self.assertEqual(fetched_topology, topology)