from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_list_services_with_backend_state(self):
    cs = fakes.FakeClient(api_version=api_versions.APIVersion('3.49'))
    services_list = cs.services.list()
    cs.assert_called('GET', '/os-services')
    self.assertEqual(3, len(services_list))
    for service in services_list:
        self.assertIsInstance(service, services.Service)
        if service.binary == 'cinder-volume':
            self.assertIsNotNone(getattr(service, 'backend_state', None))
    self._assert_request_id(services_list)