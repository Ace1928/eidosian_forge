from openstack.container_infrastructure_management.v1 import service
from openstack.tests.unit import base
def test_list_magnum_services(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(service_type='container-infrastructure-management', resource='mservices'), json=dict(mservices=[magnum_service_obj]))])
    mservices_list = self.cloud.list_magnum_services()
    self.assertEqual(mservices_list[0].to_dict(computed=False), service.Service(**magnum_service_obj).to_dict(computed=False))
    self.assert_calls()