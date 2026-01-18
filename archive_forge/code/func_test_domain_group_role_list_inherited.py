import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_group_role_list_inherited(self):
    group_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    ref_list = [self.new_ref(), self.new_ref()]
    self.stub_entity('GET', ['OS-INHERIT', 'domains', domain_id, 'groups', group_id, self.collection_key, 'inherited_to_projects'], entity=ref_list)
    returned_list = self.manager.list(domain=domain_id, group=group_id, os_inherit_extension_inherited=True)
    self.assertThat(ref_list, matchers.HasLength(len(returned_list)))
    [self.assertIsInstance(r, self.model) for r in returned_list]