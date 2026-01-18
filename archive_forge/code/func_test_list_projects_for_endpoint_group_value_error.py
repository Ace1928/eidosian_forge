import uuid
from keystoneclient.tests.unit.v3 import utils
def test_list_projects_for_endpoint_group_value_error(self):
    self.assertRaises(ValueError, self.manager.list_projects_for_endpoint_group, endpoint_group='')
    self.assertRaises(ValueError, self.manager.list_projects_for_endpoint_group, endpoint_group=None)