import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import domains
def test_list_filter_disabled(self):
    expected_query = {'enabled': '0'}
    super(DomainTests, self).test_list(expected_query=expected_query, enabled=False)