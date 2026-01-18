import uuid
from keystone import exception
def test_check_policy_association(self):
    association = self.create_association(service_id=uuid.uuid4().hex, region_id=uuid.uuid4().hex)
    self.driver.check_policy_association(**association)
    association.pop('region_id')
    self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)