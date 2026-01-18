import uuid
from keystone import exception
def test_delete_association_by_policy(self):
    policy_id = uuid.uuid4().hex
    first = self.create_association(endpoint_id=uuid.uuid4().hex, policy_id=policy_id)
    second = self.create_association(service_id=uuid.uuid4().hex, policy_id=policy_id)
    self.driver.delete_association_by_policy(policy_id)
    for association in [first, second]:
        self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)