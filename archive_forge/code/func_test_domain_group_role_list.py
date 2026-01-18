import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_group_role_list(self):
    group_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    ref_list = [self.new_ref(), self.new_ref()]
    self.stub_entity('GET', ['domains', domain_id, 'groups', group_id, self.collection_key], entity=ref_list)
    self.manager.list(domain=domain_id, group=group_id)