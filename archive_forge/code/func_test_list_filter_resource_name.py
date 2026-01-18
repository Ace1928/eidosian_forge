import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import registered_limits
def test_list_filter_resource_name(self):
    resource_name = uuid.uuid4().hex
    self.test_list(resource_name=resource_name)