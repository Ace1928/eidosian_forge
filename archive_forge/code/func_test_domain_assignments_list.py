from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_domain_assignments_list(self):
    ref_list = self.TEST_USER_DOMAIN_LIST
    self.stub_entity('GET', [self.collection_key, '?scope.domain.id=%s' % self.TEST_DOMAIN_ID], entity=ref_list)
    returned_list = self.manager.list(domain=self.TEST_DOMAIN_ID)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'scope.domain.id': self.TEST_DOMAIN_ID}
    self.assertQueryStringContains(**kwargs)