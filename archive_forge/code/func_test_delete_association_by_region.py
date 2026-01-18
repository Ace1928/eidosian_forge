import uuid
from keystone import exception
def test_delete_association_by_region(self):
    region_id = uuid.uuid4().hex
    first = self.create_association(service_id=uuid.uuid4().hex, region_id=region_id)
    second = self.create_association(service_id=uuid.uuid4().hex, region_id=region_id)
    self.driver.delete_association_by_region(region_id)
    for association in [first, second]:
        self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)