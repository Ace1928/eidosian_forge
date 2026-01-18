from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_group_assignments_list(self):
    ref_list = self.TEST_GROUP_PROJECT_LIST
    self.stub_entity('GET', [self.collection_key, '?group.id=%s' % self.TEST_GROUP_ID], entity=ref_list)
    returned_list = self.manager.list(group=self.TEST_GROUP_ID)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'group.id': self.TEST_GROUP_ID}
    self.assertQueryStringContains(**kwargs)