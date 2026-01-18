import uuid
from keystoneclient.tests.unit.v3 import utils
def test_list_endpoint_groups_for_project_value_error(self):
    for value in ('', None):
        self.assertRaises(ValueError, self.manager.list_endpoint_groups_for_project, project=value)