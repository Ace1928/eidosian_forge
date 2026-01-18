import uuid
from keystone.common import driver_hints
from keystone import exception
def test_create_group_all_attrs(self):
    group_id = uuid.uuid4().hex
    group = {'id': group_id, 'name': uuid.uuid4().hex, 'description': uuid.uuid4().hex}
    if self.driver.is_domain_aware():
        group['domain_id'] = uuid.uuid4().hex
    new_group = self.driver.create_group(group_id, group)
    self.assertEqual(group, new_group)