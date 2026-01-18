import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_list_all_servers(self):
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    server = self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, wait=True)
    found_server = False
    for s in self.operator_cloud.list_servers(all_projects=True):
        if s.name == server.name:
            found_server = True
    self.assertTrue(found_server)