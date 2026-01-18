import uuid
from keystone.common import driver_hints
from keystone import exception
def test_create_group_duplicate_exc(self):
    group1_id = uuid.uuid4().hex
    name = uuid.uuid4().hex
    domain = uuid.uuid4().hex
    group1 = {'id': group1_id, 'name': name}
    if self.driver.is_domain_aware():
        group1['domain_id'] = domain
    self.driver.create_group(group1_id, group1)
    group2_id = uuid.uuid4().hex
    group2 = {'id': group2_id, 'name': name}
    if self.driver.is_domain_aware():
        group2['domain_id'] = domain
    self.assertRaises(exception.Conflict, self.driver.create_group, group2_id, group2)