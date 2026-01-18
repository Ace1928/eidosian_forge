from openstack.tests.functional import base
def test_disable_enable(self):
    for srv in self.operator_cloud.block_storage.services():
        if srv.name == 'nova-block_storage':
            self.operator_cloud.block_storage.disable_service(srv)
            self.operator_cloud.block_storage.enable_service(srv)
            break