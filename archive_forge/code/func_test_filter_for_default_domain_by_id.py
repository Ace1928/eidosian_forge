import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import domains
def test_filter_for_default_domain_by_id(self):
    ref = self.new_ref(id='default')
    super(DomainTests, self).test_list_by_id(ref=ref, id=ref['id'])