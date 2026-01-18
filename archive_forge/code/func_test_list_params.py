import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import domain_configs
def test_list_params(self):
    self.assertRaises(exceptions.MethodNotImplemented, self.manager.list)