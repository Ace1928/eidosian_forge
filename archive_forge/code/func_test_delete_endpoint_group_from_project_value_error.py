import uuid
from keystoneclient.tests.unit.v3 import utils
def test_delete_endpoint_group_from_project_value_error(self):
    for value in ('', None):
        self.assertRaises(ValueError, self.manager.delete_endpoint_group_from_project, project=value, endpoint_group=value)
        self.assertRaises(ValueError, self.manager.delete_endpoint_group_from_project, project=uuid.uuid4().hex, endpoint_group=value)
        self.assertRaises(ValueError, self.manager.delete_endpoint_group_from_project, project=value, endpoint_group=uuid.uuid4().hex)