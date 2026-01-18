from openstack.network.v2 import network
from openstack.tests.functional import base
def test_find_with_filter(self):
    if not self.operator_cloud:
        self.skipTest('Operator cloud required for this test')
    project_id_1 = '1'
    project_id_2 = '2'
    sot1 = self.operator_cloud.network.create_network(name=self.NAME, project_id=project_id_1)
    sot2 = self.operator_cloud.network.create_network(name=self.NAME, project_id=project_id_2)
    sot = self.operator_cloud.network.find_network(self.NAME, project_id=project_id_1)
    self.assertEqual(project_id_1, sot.project_id)
    self.operator_cloud.network.delete_network(sot1.id)
    self.operator_cloud.network.delete_network(sot2.id)