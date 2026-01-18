import uuid
from keystone import exception
def test_delete_policy_association(self):
    association = self.create_association(endpoint_id=uuid.uuid4().hex)
    self.driver.delete_policy_association(**association)
    self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)