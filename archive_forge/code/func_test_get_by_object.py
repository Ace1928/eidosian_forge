from unittest import mock
import ddt
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import security_services
def test_get_by_object(self):
    security_service = self._FakeSecurityService()
    with mock.patch.object(self.manager, '_get', mock.Mock()):
        self.manager.get(security_service)
        self.manager._get.assert_called_once_with(security_services.RESOURCE_PATH % security_service.id, security_services.RESOURCE_NAME)