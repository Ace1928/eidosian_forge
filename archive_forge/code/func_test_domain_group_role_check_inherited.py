import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_group_role_check_inherited(self):
    group_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_url('HEAD', ['OS-INHERIT', 'domains', domain_id, 'groups', group_id, self.collection_key, ref['id'], 'inherited_to_projects'], status_code=204)
    self.manager.check(role=ref['id'], domain=domain_id, group=group_id, os_inherit_extension_inherited=True)