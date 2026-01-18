import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def test_manager_api_reference(self):
    manager = self._create_manager_instance()
    second_manager = self._create_manager_instance()
    self.assertIs(second_manager, getattr(manager, second_manager._provides_api))
    self.assertIs(manager, getattr(second_manager, manager._provides_api))