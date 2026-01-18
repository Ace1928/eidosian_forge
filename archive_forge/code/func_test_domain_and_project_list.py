from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_domain_and_project_list(self):
    self.assertRaises(exceptions.ValidationError, self.manager.list, domain=self.TEST_DOMAIN_ID, project=self.TEST_TENANT_ID)