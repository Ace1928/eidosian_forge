import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import limits
def test_list_filtered_by_region(self):
    region_id = uuid.uuid4().hex
    expected_query = {'region_id': region_id}
    self.test_list(expected_query=expected_query, region=region_id)