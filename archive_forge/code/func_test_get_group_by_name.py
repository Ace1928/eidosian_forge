import uuid
from keystone.common import driver_hints
from keystone import exception
def test_get_group_by_name(self):
    domain_id = uuid.uuid4().hex
    group = self.create_group(domain_id=domain_id)
    actual_group = self.driver.get_group_by_name(group['name'], domain_id)
    self.assertEqual(group['id'], actual_group['id'])